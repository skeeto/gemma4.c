// A minimal OpenAI-style chat completions server for gemma4.c.
//
// Implements just enough HTTP, SSE, and JSON for coding harnesses that speak
// the OpenAI chat completions protocol (tested against OpenCode and Pi): one
// streaming POST /chat/completions endpoint, with tool calling prompted
// through the chat template and mapped onto the standard tools/tool_calls
// wire format. Requests are served one at a time, and the KV cache carries
// over between requests that share a token prefix.

#define GEMMA4_LIB
#include "gemma4.c"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <strings.h>
#include <sys/socket.h>

#define FENCE "```tool_call"
#define MAX_BODY (64 << 20)
#define DEFAULT_MAX_NEW 4096

// ----------------------------------------------------------------------------
// Growable text buffer

typedef struct {
    char *data;
    size_t len, cap;
} Buffer;

void buffer_grow(Buffer *buffer, size_t extra) {
    if (buffer->len + extra + 1 <= buffer->cap) return;
    size_t cap = buffer->cap ? buffer->cap : 4096;
    while (cap < buffer->len + extra + 1) cap *= 2;
    buffer->data = realloc(buffer->data, cap);
    if (!buffer->data) { perror("realloc"); exit(1); }
    buffer->cap = cap;
}

void append(Buffer *buffer, const char *data, size_t len) {
    buffer_grow(buffer, len);
    memcpy(buffer->data + buffer->len, data, len);
    buffer->len += len;
    buffer->data[buffer->len] = 0;
}

void append_str(Buffer *buffer, const char *s) { append(buffer, s, strlen(s)); }

void buffer_clear(Buffer *buffer) {
    buffer->len = 0;
    if (buffer->data) buffer->data[0] = 0;
}

// Appends text with JSON string escaping applied.
void append_escaped(Buffer *buffer, const char *text, size_t len) {
    buffer_grow(buffer, len * 6);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        switch (c) {
        case '"': append_str(buffer, "\\\""); break;
        case '\\': append_str(buffer, "\\\\"); break;
        case '\n': append_str(buffer, "\\n"); break;
        case '\r': append_str(buffer, "\\r"); break;
        case '\t': append_str(buffer, "\\t"); break;
        default:
            if (c < 0x20) {
                char escape[8];
                snprintf(escape, sizeof(escape), "\\u%04x", c);
                append_str(buffer, escape);
            } else append(buffer, (const char *)&text[i], 1);
        }
    }
}

// ----------------------------------------------------------------------------
// JSON reader: cursor-based helpers over the raw request body. Values are
// referenced by their starting position; json_skip() finds where they end.

const char *json_ws(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    return s;
}

// Returns the position just past the value at s, or NULL on malformed input.
const char *json_skip(const char *s) {
    s = json_ws(s);
    if (*s == '"') {
        for (s++; *s && *s != '"'; s++)
            if (*s == '\\' && s[1]) s++;
        return *s == '"' ? s + 1 : NULL;
    }
    if (*s == '{' || *s == '[') {
        int object = *s == '{';
        char close = object ? '}' : ']';
        s = json_ws(s + 1);
        if (*s == close) return s + 1;
        for (;;) {
            if (object) {
                if (!(s = json_skip(s))) return NULL; // The member key.
                s = json_ws(s);
                if (*s++ != ':') return NULL;
            }
            if (!(s = json_skip(s))) return NULL;
            s = json_ws(s);
            if (*s == ',') { s = json_ws(s + 1); continue; }
            return *s == close ? s + 1 : NULL;
        }
    }
    if (!strncmp(s, "true", 4)) return s + 4;
    if (!strncmp(s, "false", 5)) return s + 5;
    if (!strncmp(s, "null", 4)) return s + 4;
    const char *start = s;
    while (*s && strchr("+-0123456789.eE", *s)) s++;
    return s > start ? s : NULL;
}

// Returns the value of key in the object at s, or NULL when absent.
const char *json_get(const char *s, const char *key) {
    if (!s || *(s = json_ws(s)) != '{') return NULL;
    size_t key_len = strlen(key);
    s = json_ws(s + 1);
    while (*s == '"') {
        const char *name = s + 1;
        const char *end = json_skip(s);
        if (!end) return NULL;
        int match = (size_t)(end - 1 - name) == key_len && !strncmp(name, key, key_len);
        s = json_ws(end);
        if (*s++ != ':') return NULL;
        s = json_ws(s);
        if (match) return s;
        if (!(s = json_skip(s))) return NULL;
        s = json_ws(s);
        if (*s != ',') break;
        s = json_ws(s + 1);
    }
    return NULL;
}

// Returns the first element of the array at s, or NULL when empty or absent.
const char *json_first(const char *s) {
    if (!s || *(s = json_ws(s)) != '[') return NULL;
    s = json_ws(s + 1);
    return *s == ']' ? NULL : s;
}

// Returns the element following item, or NULL at the end of the array.
const char *json_next(const char *item) {
    const char *s = json_skip(item);
    if (!s) return NULL;
    s = json_ws(s);
    return *s == ',' ? json_ws(s + 1) : NULL;
}

unsigned json_hex4(const char *s) {
    unsigned value = 0;
    for (int i = 0; i < 4; i++) {
        int c = s[i] | 32;
        value = value * 16 + (c >= 'a' ? c - 'a' + 10 : c - '0');
    }
    return value;
}

size_t utf8_encode(char *out, unsigned cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | cp >> 6); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | cp >> 12); out[1] = (char)(0x80 | (cp >> 6 & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | cp >> 18); out[1] = (char)(0x80 | (cp >> 12 & 0x3F)); out[2] = (char)(0x80 | (cp >> 6 & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Decodes the JSON string at s into out, returning its length or -1.
long json_string(const char *s, char *out, size_t cap) {
    if (!s) return -1;
    s = json_ws(s);
    if (*s++ != '"') return -1;
    size_t n = 0;
    while (*s && *s != '"') {
        if (n + 8 >= cap) return -1;
        if (*s != '\\') { out[n++] = *s++; continue; }
        switch (*++s) {
        case '"': case '\\': case '/': out[n++] = *s++; break;
        case 'b': out[n++] = '\b'; s++; break;
        case 'f': out[n++] = '\f'; s++; break;
        case 'n': out[n++] = '\n'; s++; break;
        case 'r': out[n++] = '\r'; s++; break;
        case 't': out[n++] = '\t'; s++; break;
        case 'u': {
            if (!s[1] || !s[2] || !s[3] || !s[4]) return -1;
            unsigned cp = json_hex4(s + 1);
            s += 5;
            if (cp >= 0xD800 && cp < 0xDC00 && s[0] == '\\' && s[1] == 'u' && s[2] && s[3] && s[4] && s[5]) {
                unsigned low = json_hex4(s + 2);
                if (low >= 0xDC00 && low < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00); s += 6; }
            }
            n += utf8_encode(out + n, cp);
            break;
        }
        default: return -1;
        }
    }
    out[n] = 0;
    return *s == '"' ? (long)n : -1;
}

// True when the JSON string at s equals the literal.
int json_streq(const char *s, const char *literal) {
    char decoded[64];
    return s && json_string(s, decoded, sizeof(decoded)) >= 0 && !strcmp(decoded, literal);
}

double json_number(const char *s, double fallback) {
    if (!s) return fallback;
    char *end;
    double value = strtod(json_ws(s), &end);
    return end == json_ws(s) ? fallback : value;
}

// Decodes the JSON string at value and appends it to the buffer.
void append_json_decoded(Buffer *buffer, const char *value) {
    if (!value || *(value = json_ws(value)) != '"') return;
    const char *end = json_skip(value);
    if (!end) return;
    buffer_grow(buffer, (size_t)(end - value) + 16);
    long n = json_string(value, buffer->data + buffer->len, buffer->cap - buffer->len);
    if (n > 0) buffer->len += (size_t)n;
    buffer->data[buffer->len] = 0;
}

// Decodes a message's content, which is a string, null, or an array of parts.
void decode_content(const char *value, Buffer *out) {
    buffer_clear(out);
    if (!value) return;
    value = json_ws(value);
    if (*value == '"') append_json_decoded(out, value);
    else if (*value == '[')
        for (const char *part = json_first(value); part; part = json_next(part)) {
            const char *type = json_get(part, "type");
            if (!type || json_streq(type, "text")) append_json_decoded(out, json_get(part, "text"));
        }
}

// ----------------------------------------------------------------------------
// SSE responses

typedef struct {
    int socket;
    int failed; // The client hung up, so stop generating for it.
} Client;

int call_counter;

int send_all(Client *client, const char *data, size_t len) {
    while (len && !client->failed) {
        ssize_t sent = write(client->socket, data, len);
        if (sent < 0 && errno == EINTR) continue;
        if (sent < 0) { perror("write"); client->failed = 1; return -1; }
        data += sent;
        len -= (size_t)sent;
    }
    return client->failed ? -1 : 0;
}

void http_error(Client *client, int status, const char *reason, const char *message) {
    char body[512], head[256];
    int body_len = snprintf(body, sizeof(body), "{\"error\":{\"message\":\"%s\"}}", message);
    int head_len = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
        status, reason, body_len);
    send_all(client, head, (size_t)head_len);
    send_all(client, body, (size_t)body_len);
}

// Sends one chat.completion.chunk SSE frame. finish and usage may be NULL.
void send_chunk(Client *client, const char *delta, const char *finish, const char *usage) {
    static Buffer frame;
    buffer_clear(&frame);
    append_str(&frame, "data: {\"id\":\"gemma4\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"gemma4\",\"choices\":[{\"index\":0,\"delta\":");
    append_str(&frame, delta);
    append_str(&frame, ",\"finish_reason\":");
    if (finish) { append_str(&frame, "\""); append_str(&frame, finish); append_str(&frame, "\""); }
    else append_str(&frame, "null");
    append_str(&frame, "}]");
    if (usage) { append_str(&frame, ",\"usage\":"); append_str(&frame, usage); }
    append_str(&frame, "}\n\n");
    send_all(client, frame.data, frame.len);
}

void send_content(Client *client, const char *text, size_t len) {
    if (!len || client->failed) return;
    static Buffer delta;
    buffer_clear(&delta);
    append_str(&delta, "{\"content\":\"");
    append_escaped(&delta, text, len);
    append_str(&delta, "\"}");
    send_chunk(client, delta.data, NULL, NULL);
}

// Sends a complete tool call in a single delta, which both harnesses accept.
void send_tool_call(Client *client, const char *name, const char *arguments, size_t arguments_len) {
    static Buffer delta;
    buffer_clear(&delta);
    char id[32];
    snprintf(id, sizeof(id), "call_%d", call_counter++);
    append_str(&delta, "{\"tool_calls\":[{\"index\":0,\"id\":\"");
    append_str(&delta, id);
    append_str(&delta, "\",\"type\":\"function\",\"function\":{\"name\":\"");
    append_escaped(&delta, name, strlen(name));
    append_str(&delta, "\",\"arguments\":\"");
    append_escaped(&delta, arguments, arguments_len);
    append_str(&delta, "\"}}]}");
    send_chunk(client, delta.data, NULL, NULL);
}

void send_finish(Client *client, const char *finish, int prompt, int cached, int completion) {
    char usage[192];
    snprintf(usage, sizeof(usage),
        "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}",
        prompt, completion, prompt + completion, cached);
    send_chunk(client, "{}", finish, usage);
    send_all(client, "data: [DONE]\n\n", 15);
}

// ----------------------------------------------------------------------------
// Prompt construction

int token_ids[MAX_CONTEXT];
int scratch_ids[MAX_CONTEXT];
int tokens_total;
int prev_ids[MAX_CONTEXT]; // Mirrors the token positions held by the KV cache.
int prev_count;
FILE *debug_file; // Set by GEMMA4_DEBUG=path to record request bodies and rendered prompts.

// Appends one templated turn to the token sequence; fails on overflow.
int push_turn(const Tokenizer *tokenizer, const char *prefix, const char *text, const char *suffix) {
    if (debug_file) fprintf(debug_file, "%s%s%s", prefix, text ? text : "", suffix);
    const char *segments[3] = {prefix, text ? text : "", suffix};
    int count = tokenize(tokenizer, segments, scratch_ids, MAX_CONTEXT);
    if (count < 0 || tokens_total + count - 1 > MAX_CONTEXT) return 0;
    memcpy(token_ids + tokens_total, scratch_ids + 1, (count - 1) * sizeof(int)); // Skip the <bos> tokenize() prepends.
    tokens_total += count - 1;
    return 1;
}

// Renders the tool schemas and calling convention into system text.
void render_tools(Buffer *out, const char *tools) {
    append_str(out, "You can call tools. The available tools:\n\n");
    for (const char *tool = json_first(tools); tool; tool = json_next(tool)) {
        const char *function = json_get(tool, "function");
        if (!function) continue;
        append_str(out, "## ");
        append_json_decoded(out, json_get(function, "name"));
        append_str(out, "\n");
        append_json_decoded(out, json_get(function, "description"));
        const char *parameters = json_get(function, "parameters");
        const char *end = parameters ? json_skip(parameters) : NULL;
        if (end) {
            append_str(out, "\nParameters (JSON schema): ");
            append(out, parameters, (size_t)(end - parameters));
        }
        append_str(out, "\n\n");
    }
    append_str(out,
        "To call a tool, end your reply with exactly one block of this form and write nothing after it:\n"
        "```tool_call\n{\"name\": \"<tool name>\", \"arguments\": {<arguments matching the schema>}}\n```\n"
        "The tool result will arrive in a <tool_response> block. "
        "Reply in plain text when no tool is needed.\n");
}

// Builds the full token sequence for the conversation; returns an error string or NULL.
const char *build_prompt(const Tokenizer *tokenizer, const char *body) {
    static Buffer content, system_text;
    const char *messages = json_get(body, "messages");
    if (!json_first(messages)) return "request has no messages";

    tokens_total = 1;
    token_ids[0] = 2; // <bos>

    // System text and the tool preamble form one leading user turn.
    buffer_clear(&system_text);
    for (const char *msg = json_first(messages); msg; msg = json_next(msg)) {
        const char *role = json_get(msg, "role");
        if (json_streq(role, "system") || json_streq(role, "developer")) {
            decode_content(json_get(msg, "content"), &content);
            append(&system_text, content.data ? content.data : "", content.len);
            append_str(&system_text, "\n\n");
        }
    }
    const char *tools = json_get(body, "tools");
    if (json_first(tools)) render_tools(&system_text, tools);
    if (system_text.len && !push_turn(tokenizer, "<|turn>user\n", system_text.data, "<turn|>\n"))
        return "prompt exceeds the context window: context_length_exceeded";

    for (const char *msg = json_first(messages); msg; msg = json_next(msg)) {
        const char *role = json_get(msg, "role");
        int ok = 1;
        if (json_streq(role, "user")) {
            decode_content(json_get(msg, "content"), &content);
            ok = push_turn(tokenizer, "<|turn>user\n", content.data, "<turn|>\n");
        } else if (json_streq(role, "tool")) {
            static Buffer wrapped;
            buffer_clear(&wrapped);
            append_str(&wrapped, "<tool_response id=\"");
            append_json_decoded(&wrapped, json_get(msg, "tool_call_id"));
            append_str(&wrapped, "\">\n");
            decode_content(json_get(msg, "content"), &content);
            append(&wrapped, content.data ? content.data : "", content.len);
            append_str(&wrapped, "\n</tool_response>");
            ok = push_turn(tokenizer, "<|turn>user\n", wrapped.data, "<turn|>\n");
        } else if (json_streq(role, "assistant")) {
            decode_content(json_get(msg, "content"), &content);
            // Replay earlier tool calls in the same fenced form the model emitted.
            for (const char *call = json_first(json_get(msg, "tool_calls")); call; call = json_next(call)) {
                const char *function = json_get(call, "function");
                if (!function) continue;
                if (content.len && content.data[content.len - 1] != '\n') append_str(&content, "\n");
                append_str(&content, "```tool_call\n{\"name\": \"");
                append_json_decoded(&content, json_get(function, "name"));
                append_str(&content, "\", \"arguments\": ");
                size_t before = content.len;
                append_json_decoded(&content, json_get(function, "arguments"));
                if (content.len == before) append_str(&content, "{}");
                append_str(&content, "}\n```\n");
            }
            ok = push_turn(tokenizer, "<|turn>model\n", content.data, "<turn|>\n");
        }
        if (!ok) return "prompt exceeds the context window: context_length_exceeded";
    }
    if (!push_turn(tokenizer, "<|turn>model\n", "", "")) // Cue the model's reply.
        return "prompt exceeds the context window: context_length_exceeded";
    return NULL;
}

// ----------------------------------------------------------------------------
// Generation with fence holdback

typedef struct {
    Client *client;
    Buffer pending; // Text withheld because it may open a tool_call fence.
    Buffer tool;    // Text captured after the fence marker.
    int in_tool;
    int tool_done;
} Stream;

void stream_check_close(Stream *stream) {
    char *close = strstr(stream->tool.data ? stream->tool.data : "", "\n```");
    if (!close) return;
    stream->tool.len = (size_t)(close - stream->tool.data);
    stream->tool.data[stream->tool.len] = 0;
    stream->tool_done = 1;
}

void stream_text(Stream *stream, const char *text) {
    if (stream->in_tool) {
        append_str(&stream->tool, text);
        stream_check_close(stream);
        return;
    }
    append_str(&stream->pending, text);
    char *fence = strstr(stream->pending.data, FENCE);
    if (fence) {
        send_content(stream->client, stream->pending.data, (size_t)(fence - stream->pending.data));
        stream->in_tool = 1;
        append_str(&stream->tool, fence + strlen(FENCE));
        buffer_clear(&stream->pending);
        stream_check_close(stream);
        return;
    }
    // Flush all but the longest tail that could still begin the fence marker.
    size_t keep = 0;
    for (size_t k = strlen(FENCE) - 1; k > 0; k--)
        if (stream->pending.len >= k && !strncmp(stream->pending.data + stream->pending.len - k, FENCE, k)) { keep = k; break; }
    if (stream->pending.len > keep) {
        send_content(stream->client, stream->pending.data, stream->pending.len - keep);
        memmove(stream->pending.data, stream->pending.data + stream->pending.len - keep, keep);
        stream->pending.len = keep;
        stream->pending.data[keep] = 0;
    }
}

// Extracts name and the raw arguments object from the fenced block.
int parse_tool_call(const char *text, char *name, size_t name_cap, const char **arguments, size_t *arguments_len) {
    if (json_string(json_get(text, "name"), name, name_cap) < 0) return 0;
    const char *value = json_get(text, "arguments");
    const char *end = value ? json_skip(value) : NULL;
    if (end) { *arguments = value; *arguments_len = (size_t)(end - value); }
    else { *arguments = "{}"; *arguments_len = 2; }
    return 1;
}

// ----------------------------------------------------------------------------
// Request handling

void handle_completion(Model *model, InferenceState *state, Client *client, const char *body) {
    const char *error = build_prompt(&model->tokenizer, body);
    if (error) { http_error(client, 400, "Bad Request", error); return; }
    int prompt_tokens = tokens_total;
    float temperature = (float)json_number(json_get(body, "temperature"), 0.0);
    const char *max_field = json_get(body, "max_tokens");
    if (!max_field) max_field = json_get(body, "max_completion_tokens");
    int max_new = (int)json_number(max_field, DEFAULT_MAX_NEW);
    if (max_new <= 0) max_new = DEFAULT_MAX_NEW;

    const char *header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    send_all(client, header, strlen(header));

    // Reuse the KV cache for the tokens shared with the previous request,
    // keeping at least one token to produce logits for the last position.
    int common = 0;
    while (common < prev_count && common < prompt_tokens - 1 && prev_ids[common] == token_ids[common]) common++;
    // The sliding caches retain only the newest 1024 positions, so start over
    // when re-prefilling from common would read evicted positions.
    if (prev_count > SLIDING_WINDOW + BATCH_SIZE && common - (SLIDING_WINDOW - 1) < prev_count - (SLIDING_WINDOW + BATCH_SIZE))
        common = 0;
    // Align the first chunk to the batch grid: sliding-cache writes are
    // contiguous, so a chunk starting off the grid could cross the wrap boundary.
    int fed = prompt_tokens - common;
    int head = (BATCH_SIZE - common % BATCH_SIZE) % BATCH_SIZE;
    if (head > fed) head = fed;
    if (head) forward(model, state, token_ids + common, head, common);
    prefill(model, state, token_ids + common + head, fed - head, common + head, 0);
    int last_index = fed <= head ? fed - 1 : (fed - head - 1) % BATCH_SIZE;

    if (debug_file) {
        fprintf(debug_file, "prompt ids (common=%d):", common);
        for (int i = 0; i < prompt_tokens; i++) fprintf(debug_file, " %d", token_ids[i]);
        fprintf(debug_file, "\ngenerated ids:");
    }
    Stream stream = {.client = client};
    buffer_clear(&stream.pending);
    buffer_clear(&stream.tool);
    const char *finish = "length";
    int position = prompt_tokens;
    int end = prompt_tokens + max_new < MAX_CONTEXT ? prompt_tokens + max_new : MAX_CONTEXT;
    for (int first = 1; position < end && !client->failed; first = 0) {
        int next = sample(logits(model, state, first ? last_index : 0), VOCAB_SIZE, temperature);
        if (next == 1 || next == 106) { finish = "stop"; break; } // <eos> or <turn|>.
        forward(model, state, &next, 1, position);
        token_ids[position++] = next;
        if (debug_file) fprintf(debug_file, " %d", next);
        stream_text(&stream, token_text(&model->tokenizer, next));
        if (stream.tool_done) { finish = "stop"; break; }
    }
    prev_count = position;
    memcpy(prev_ids, token_ids, (size_t)position * sizeof(int));

    if (stream.in_tool) {
        char name[128];
        const char *arguments;
        size_t arguments_len;
        if (parse_tool_call(stream.tool.data, name, sizeof(name), &arguments, &arguments_len)) {
            send_tool_call(client, name, arguments, arguments_len);
            finish = "tool_calls";
        } else { // Not a valid call after all: emit the withheld text as content.
            send_content(client, FENCE, strlen(FENCE));
            send_content(client, stream.tool.data, stream.tool.len);
        }
    } else send_content(client, stream.pending.data, stream.pending.len);
    send_finish(client, finish, prompt_tokens, common, position - prompt_tokens);

    fprintf(stderr, "%d prompt tokens (%d cached), %d generated, %s\n",
            prompt_tokens, common, position - prompt_tokens, finish);
    free(stream.pending.data);
    free(stream.tool.data);
}

void handle_connection(Model *model, InferenceState *state, int socket_fd) {
    Client client = {.socket = socket_fd};
    static Buffer request;
    buffer_clear(&request);
    char *split = NULL;
    while (!split) {
        char data[16384];
        ssize_t received = read(socket_fd, data, sizeof(data));
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0 || request.len > MAX_BODY) return;
        append(&request, data, (size_t)received);
        split = strstr(request.data, "\r\n\r\n");
    }
    size_t body_offset = (size_t)(split - request.data) + 4;

    *split = 0; // Terminate the headers for parsing.
    char *line_end = strchr(request.data, '\r');
    if (line_end) *line_end = 0;
    int matched = !strncmp(request.data, "POST ", 5) && strstr(request.data, "/chat/completions ");
    long content_length = -1;
    for (const char *h = line_end ? line_end + 1 : split; (h = strchr(h, '\n')); )
        if (!strncasecmp(++h, "Content-Length:", 15)) { content_length = strtol(h + 15, NULL, 10); break; }

    if (!matched) { http_error(&client, 404, "Not Found", "only POST /chat/completions is served"); return; }
    if (content_length < 2 || content_length > MAX_BODY) { http_error(&client, 400, "Bad Request", "missing or oversized body"); return; }
    while (request.len < body_offset + (size_t)content_length) {
        char data[65536];
        ssize_t received = read(socket_fd, data, sizeof(data));
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) return;
        append(&request, data, (size_t)received);
    }
    if (debug_file) {
        fprintf(debug_file, "==== request body ====\n%s\n==== rendered prompt ====\n", request.data + body_offset);
    }
    handle_completion(model, state, &client, request.data + body_offset);
    if (debug_file) { fprintf(debug_file, "\n==== end ====\n"); fflush(debug_file); }
}

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    int port = 8080;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else { fprintf(stderr, "usage: server [-m model] [-p port]\n"); return 1; }
    }
    Model *model = load_model(model_path);
    if (!model) return 1;
    InferenceState *state = calloc(1, sizeof(*state));
    rng_state = (unsigned long long)(time_seconds() * 1e9);
    if (getenv("GEMMA4_DEBUG")) debug_file = fopen(getenv("GEMMA4_DEBUG"), "w");

    signal(SIGPIPE, SIG_IGN);
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) || listen(listener, 8)) {
        perror("bind");
        return 1;
    }
    fprintf(stderr, "listening on http://127.0.0.1:%d/v1/chat/completions\n", port);

    for (;;) {
        int socket_fd = accept(listener, NULL, NULL);
        if (socket_fd < 0) continue;
        handle_connection(model, state, socket_fd);
        // A graceful shutdown and drain keeps the tail of the response from
        // being discarded by an abrupt close.
        shutdown(socket_fd, SHUT_WR);
        struct timeval drain_timeout = {.tv_sec = 5};
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &drain_timeout, sizeof(drain_timeout));
        for (char drain[4096];;) {
            ssize_t received = read(socket_fd, drain, sizeof(drain));
            if (received < 0 && errno == EINTR) continue;
            if (received <= 0) break;
        }
        close(socket_fd);
    }
}

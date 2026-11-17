// A minimal OpenAI-style chat completions server for gemma4.c.
//
// Implements just enough HTTP, SSE, and JSON for coding harnesses that speak
// the OpenAI chat completions protocol (tested against OpenCode, Pi, and
// Prospero): one streaming POST /chat/completions endpoint, with tool calling
// prompted through the chat template and mapped onto the standard
// tools/tool_calls wire format, and GET /models listing the one model.
// Requests are served one at a time, and the KV cache carries over between
// requests that share a token prefix.

#define GEMMA4_LIB
#include "gemma4.c"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <strings.h>
#include <sys/socket.h>

#define MAX_BODY (64 << 20)
#define DEFAULT_MAX_NEW 4096
#define MODEL_ID "gemma4"
#define KEEPALIVE_SECONDS 5.0

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
    int failed;         // The client hung up, so stop generating for it.
    double last_write;  // When bytes last went out, for keep-alive pings.
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
    client->last_write = time_seconds();
    return client->failed ? -1 : 0;
}

// Sends an SSE comment after a stretch of silence (a long prefill, or a
// withheld tool call) so that clients with idle timeouts keep waiting, and
// so that a client that hung up is noticed.
void keepalive(Client *client) {
    static const char ping[] = ": keepalive\n\n";
    if (time_seconds() - client->last_write >= KEEPALIVE_SECONDS)
        send_all(client, ping, sizeof(ping) - 1);
}

void send_json(Client *client, const char *body) {
    char head[256];
    int head_len = snprintf(head, sizeof(head),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
        strlen(body));
    send_all(client, head, (size_t)head_len);
    send_all(client, body, strlen(body));
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
    append_str(&frame, "data: {\"id\":\"gemma4\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"" MODEL_ID "\",\"choices\":[{\"index\":0,\"delta\":");
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
void send_tool_call(Client *client, int index, const char *name, const char *arguments, size_t arguments_len) {
    static Buffer delta;
    buffer_clear(&delta);
    char id[32];
    snprintf(id, sizeof(id), "%d", index);
    append_str(&delta, "{\"tool_calls\":[{\"index\":");
    append_str(&delta, id);
    snprintf(id, sizeof(id), "call_%d", call_counter++);
    append_str(&delta, ",\"id\":\"");
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
    static const char done[] = "data: [DONE]\n\n";
    send_all(client, done, sizeof(done) - 1);
}

// ----------------------------------------------------------------------------
// Prompt construction, following Gemma 4's own chat template
//
// The template renders the system text, thinking switch, and tool
// declarations into a system turn, and tool calls and their results into
// the model's turn with dedicated control tokens:
//
//   <|turn>system\n<|think|>\nSYSTEM<|tool>declaration:NAME{...}<tool|><turn|>\n
//   <|turn>user\nTEXT<turn|>\n
//   <|turn>model\n<|channel>thought\n...<channel|><|tool_call>call:NAME{k:v}<tool_call|>
//   <|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>TEXT<turn|>\n
//
// Strings in that syntax are delimited by the <|"|> token rather than quoted.

#define QUOTE "<|\"|>"

enum { TOK_EOS = 1, TOK_TOOL_CALL = 48, TOK_TOOL_CALL_END = 49, TOK_TOOL_RESPONSE = 50,
       TOK_QUOTE = 52, TOK_CHANNEL = 100, TOK_CHANNEL_END = 101, TOK_TURN_END = 106 };

int token_ids[MAX_CONTEXT];
int tokens_total;
int prev_ids[MAX_CONTEXT]; // Mirrors the token positions held by the KV cache.
int prev_count;
FILE *debug_file; // Set by GEMMA4_DEBUG=path to record request bodies and rendered prompts.

// The prompt as text, with a parallel mask marking the bytes where a control
// token may begin. Message contents are raw, so text that merely looks like a
// control token (a source file read by a tool, say) is tokenized as text.
typedef struct {
    Buffer text, mask;
} Prompt;

void put(Prompt *p, const char *s, size_t len, char syntax) {
    append(&p->text, s, len);
    buffer_grow(&p->mask, len);
    memset(p->mask.data + p->mask.len, syntax, len);
    p->mask.len += len;
}

void syn(Prompt *p, const char *s) { put(p, s, strlen(s), 1); }
void raw(Prompt *p, const char *s) { put(p, s, strlen(s), 0); }

// Appends raw text with surrounding whitespace removed, as Jinja's trim.
void raw_trimmed(Prompt *p, const char *s, size_t len) {
    while (len && strchr(" \t\r\n", *s)) { s++; len--; }
    while (len && strchr(" \t\r\n", s[len - 1])) len--;
    put(p, s, len, 0);
}

void quoted(Prompt *p, const char *s, size_t len) {
    syn(p, QUOTE);
    put(p, s, len, 0);
    syn(p, QUOTE);
}

// Appends the decoded JSON string at value as a <|"|>-delimited string.
void quoted_json(Prompt *p, const char *value) {
    Buffer decoded = {0};
    append_json_decoded(&decoded, value);
    quoted(p, decoded.data ? decoded.data : "", decoded.len);
    free(decoded.data);
}

// Splits text at control tokens and applies BPE to each run between them,
// as the reference tokenizer does. Returns the token count, or -1 on overflow.
int tokenize_prompt(const Tokenizer *tokenizer, const Prompt *p, int *tokens, int capacity) {
    const char *text = p->text.data;
    size_t len = p->text.len;
    int count = 0, run = 0; // run: where the current run of plain text began
    for (size_t i = 0; i < len;) {
        if (count + 4 >= capacity) return -1;
        if (p->mask.data[i] && text[i] == '<') {
            int special = -1;
            size_t special_len = 0;
            for (int k = 0; k < tokenizer->special_count; k++) {
                size_t n = strlen(tokenizer->specials[k].token);
                if (n > special_len && n <= len - i && !strncmp(text + i, tokenizer->specials[k].token, n)) {
                    special = tokenizer->specials[k].id;
                    special_len = n;
                }
            }
            if (special >= 0) {
                count = run + apply_bpe_merges(tokenizer, tokens + run, count - run);
                tokens[count++] = special;
                run = count;
                i += special_len;
                continue;
            }
        }
        char piece[8] = {0};
        if (text[i] == ' ') { memcpy(piece, "\xE2\x96\x81", 3); i++; } // SentencePiece represents spaces with U+2581.
        else {
            piece[0] = text[i++];
            if ((piece[0] & 0xC0) == 0xC0)
                for (int k = 1; k < 4 && i < len && (text[i] & 0xC0) == 0x80; k++) piece[k] = text[i++];
        }
        const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab, tokenizer->encode_vocab_count,
                                           sizeof(LookupEntry), compare_lookup_keys);
        if (entry) { tokens[count++] = entry->result; continue; }
        for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++)
            tokens[count++] = 238 + *byte; // Byte tokens occupy IDs 238 through 493.
    }
    return run + apply_bpe_merges(tokenizer, tokens + run, count - run);
}

// One member of a JSON object.
typedef struct {
    char name[128];
    const char *value;
} Member;

int compare_members(const void *a, const void *b) {
    return strcmp(((const Member *)a)->name, ((const Member *)b)->name);
}

// Lists up to cap members of the object at obj, sorted by name when asked
// (the template's dictsort), and returns how many.
int json_members(const char *obj, Member *out, int cap, int sorted) {
    if (!obj || *(obj = json_ws(obj)) != '{') return 0;
    const char *s = json_ws(obj + 1);
    int n = 0;
    while (*s == '"' && n < cap) {
        const char *end = json_skip(s);
        if (!end || json_string(s, out[n].name, sizeof(out[n].name)) < 0) break;
        s = json_ws(end);
        if (*s++ != ':') break;
        out[n++].value = json_ws(s);
        if (!(s = json_skip(s))) break;
        s = json_ws(s);
        if (*s != ',') break;
        s = json_ws(s + 1);
    }
    if (sorted) qsort(out, (size_t)n, sizeof(*out), compare_members);
    return n;
}

int json_is(const char *value, char c) { return value && *json_ws(value) == c; }

// Renders a JSON value in the template's syntax (its format_argument macro).
void put_value(Prompt *p, const char *value, int escape_keys, int sorted) {
    value = json_ws(value);
    if (*value == '"') {
        quoted_json(p, value);
    } else if (*value == '{') {
        Member members[64];
        int n = json_members(value, members, 64, sorted);
        syn(p, "{");
        for (int i = 0; i < n; i++) {
            if (i) syn(p, ",");
            if (escape_keys) quoted(p, members[i].name, strlen(members[i].name));
            else raw(p, members[i].name);
            syn(p, ":");
            put_value(p, members[i].value, escape_keys, sorted);
        }
        syn(p, "}");
    } else if (*value == '[') {
        syn(p, "[");
        for (const char *item = json_first(value); item; item = json_next(item)) {
            if (item != json_first(value)) syn(p, ",");
            put_value(p, item, escape_keys, sorted);
        }
        syn(p, "]");
    } else {
        const char *end = json_skip(value);
        put(p, value, end ? (size_t)(end - value) : 0, 0);
    }
}

// The schema's type, upper-cased; for a list of types, the first not null.
void schema_type(const char *type, char *out, size_t cap) {
    out[0] = 0;
    if (json_is(type, '[')) {
        for (const char *t = json_first(type); t; t = json_next(t))
            if (!json_streq(t, "null")) { type = t; break; }
    }
    if (json_string(type, out, cap) < 0) out[0] = 0;
    for (char *c = out; *c; c++) if (*c >= 'a' && *c <= 'z') *c -= 32;
}

// Renders a schema's properties (the template's format_parameters macro).
void put_parameters(Prompt *p, const char *properties) {
    Member members[64];
    int n = json_members(properties, members, 64, 1);
    for (int i = 0; i < n; i++) {
        const char *value = members[i].value;
        int comma = 0;
#define SEPARATE() do { if (comma) syn(p, ","); comma = 1; } while (0)
        if (i) syn(p, ",");
        raw(p, members[i].name);
        syn(p, ":{");
        const char *description = json_get(value, "description");
        if (json_is(description, '"')) { syn(p, "description:"); quoted_json(p, description); comma = 1; }
        char type[32];
        schema_type(json_get(value, "type"), type, sizeof(type));
        if (!strcmp(type, "STRING")) {
            const char *options = json_get(value, "enum");
            if (json_first(options)) { SEPARATE(); syn(p, "enum:"); put_value(p, options, 1, 1); }
        } else if (!strcmp(type, "ARRAY")) {
            Member items[16];
            int item_count = json_members(json_get(value, "items"), items, 16, 1);
            if (item_count) {
                SEPARATE();
                syn(p, "items:{");
                int first = 1;
                for (int k = 0; k < item_count; k++) {
                    const char *item = items[k].value;
                    if (!strncmp(item, "null", 4)) continue;
                    if (!first) syn(p, ",");
                    first = 0;
                    if (!strcmp(items[k].name, "properties")) {
                        syn(p, "properties:{");
                        put_parameters(p, item);
                        syn(p, "}");
                    } else if (!strcmp(items[k].name, "type")) {
                        char item_type[32];
                        schema_type(item, item_type, sizeof(item_type));
                        syn(p, "type:");
                        quoted(p, item_type, strlen(item_type));
                    } else {
                        raw(p, items[k].name);
                        syn(p, ":");
                        put_value(p, item, 1, 1);
                    }
                }
                syn(p, "}");
            }
        }
        const char *nullable = json_get(value, "nullable");
        if (nullable && !strncmp(json_ws(nullable), "true", 4)) { SEPARATE(); syn(p, "nullable:true"); }
        if (!strcmp(type, "OBJECT")) {
            const char *nested = json_get(value, "properties");
            if (json_is(nested, '{')) { SEPARATE(); syn(p, "properties:{"); put_parameters(p, nested); syn(p, "}"); }
            const char *required = json_get(value, "required");
            if (json_first(required)) { SEPARATE(); syn(p, "required:"); put_value(p, required, 1, 0); }
        }
        SEPARATE();
        syn(p, "type:");
        quoted(p, type, strlen(type));
        syn(p, "}");
#undef SEPARATE
    }
}

// Renders one tool declaration (the template's format_function_declaration).
void put_tool(Prompt *p, const char *tool) {
    const char *function = json_get(tool, "function");
    if (!function) return;
    syn(p, "<|tool>declaration:");
    Buffer name = {0};
    append_json_decoded(&name, json_get(function, "name"));
    raw(p, name.data ? name.data : "");
    free(name.data);
    syn(p, "{description:");
    const char *description = json_get(function, "description");
    if (json_is(description, '"')) quoted_json(p, description);
    else quoted(p, "", 0);
    const char *parameters = json_get(function, "parameters");
    if (json_is(parameters, '{')) {
        syn(p, ",parameters:{");
        const char *properties = json_get(parameters, "properties");
        Member probe[1];
        if (json_members(properties, probe, 1, 0)) {
            syn(p, "properties:{");
            put_parameters(p, properties);
            syn(p, "},");
        }
        const char *required = json_get(parameters, "required");
        if (json_first(required)) { syn(p, "required:"); put_value(p, required, 1, 0); syn(p, ","); }
        char type[32];
        schema_type(json_get(parameters, "type"), type, sizeof(type));
        syn(p, "type:");
        quoted(p, type[0] ? type : "OBJECT", strlen(type[0] ? type : "OBJECT"));
        syn(p, "}");
    }
    syn(p, "}<tool|>");
}

// Whether the request asks for thinking: chat_template_kwargs.enable_thinking
// if given, else any reasoning_effort but "none".
int request_thinking(const char *body) {
    const char *enable = json_get(json_get(body, "chat_template_kwargs"), "enable_thinking");
    if (enable) return !strncmp(json_ws(enable), "true", 4);
    char effort[32];
    return json_string(json_get(body, "reasoning_effort"), effort, sizeof(effort)) >= 0 && strcmp(effort, "none");
}

int role_is(const char *msg, const char *a, const char *b) {
    const char *role = json_get(msg, "role");
    return json_streq(role, a) || (b && json_streq(role, b));
}

// Builds the token sequence for the conversation. Returns an error string or
// NULL, and whether generation begins inside a thought channel.
const char *build_prompt(const Tokenizer *tokenizer, const char *body, int thinking, int *in_thought) {
    static Prompt p;
    static Buffer content;
    buffer_clear(&p.text);
    buffer_clear(&p.mask);
    *in_thought = 0;

    int count = 0, cap = 0;
    const char **messages = NULL;
    for (const char *msg = json_first(json_get(body, "messages")); msg; msg = json_next(msg)) {
        if (count == cap) {
            cap = cap ? cap * 2 : 256;
            messages = realloc(messages, (size_t)cap * sizeof(*messages));
            if (!messages) { perror("realloc"); exit(1); }
        }
        messages[count++] = msg;
    }
    if (!count) return "request has no messages";

    syn(&p, "<bos>");
    const char *tools = json_get(body, "tools");
    int start = role_is(messages[0], "system", "developer");
    if (thinking || json_first(tools) || start) {
        syn(&p, "<|turn>system\n");
        if (thinking) syn(&p, "<|think|>\n");
        if (start) {
            decode_content(json_get(messages[0], "content"), &content);
            raw_trimmed(&p, content.data ? content.data : "", content.len);
        }
        for (const char *tool = json_first(tools); tool; tool = json_next(tool)) put_tool(&p, tool);
        syn(&p, "<turn|>\n");
    }

    int last_user = -1;
    for (int i = start; i < count; i++)
        if (role_is(messages[i], "user", NULL)) last_user = i;

    int previous_assistant = 0, open_turn = 0;
    for (int i = start; i < count; i++) {
        const char *msg = messages[i];
        if (role_is(msg, "tool", NULL)) continue; // Rendered with the call that asked for it.
        if (!role_is(msg, "assistant", NULL)) {
            syn(&p, role_is(msg, "user", NULL) ? "<|turn>user\n" : "<|turn>system\n");
            decode_content(json_get(msg, "content"), &content);
            raw_trimmed(&p, content.data ? content.data : "", content.len);
            syn(&p, "<turn|>\n");
            previous_assistant = 0;
            continue;
        }

        if (!previous_assistant) syn(&p, "<|turn>model\n");
        previous_assistant = 1;
        // Thinking is kept only for the replies since the last user message.
        const char *thought = json_get(msg, "reasoning_content");
        if (!thought) thought = json_get(msg, "reasoning");
        if (thought && i > last_user) {
            decode_content(thought, &content);
            if (content.len) {
                syn(&p, "<|channel>thought\n");
                raw_trimmed(&p, content.data, content.len);
                syn(&p, "\n<channel|>");
            }
        }
        // Unlike the template, text goes ahead of the calls, where the model
        // wrote it, so that the KV cache matches what was generated.
        decode_content(json_get(msg, "content"), &content);
        raw_trimmed(&p, content.data ? content.data : "", content.len);

        const char *calls = json_get(msg, "tool_calls");
        for (const char *call = json_first(calls); call; call = json_next(call)) {
            const char *function = json_get(call, "function");
            syn(&p, "<|tool_call>call:");
            Buffer name = {0}, arguments = {0};
            append_json_decoded(&name, json_get(function, "name"));
            raw(&p, name.data ? name.data : "");
            append_json_decoded(&arguments, json_get(function, "arguments"));
            put_value(&p, arguments.data && json_is(arguments.data, '{') ? arguments.data : "{}", 0, 0);
            syn(&p, "<tool_call|>");
            free(name.data);
            free(arguments.data);
        }

        int responses = 0, next = i + 1;
        for (; next < count && role_is(messages[next], "tool", NULL); next++) {
            // The tool's name, by the id of the call it answers.
            Buffer id = {0}, name = {0};
            append_json_decoded(&id, json_get(messages[next], "tool_call_id"));
            for (const char *call = json_first(calls); call; call = json_next(call)) {
                Buffer call_id = {0};
                append_json_decoded(&call_id, json_get(call, "id"));
                if (id.data && call_id.data && !strcmp(id.data, call_id.data))
                    append_json_decoded(&name, json_get(json_get(call, "function"), "name"));
                free(call_id.data);
            }
            syn(&p, "<|tool_response>response:");
            raw(&p, name.data ? name.data : "unknown");
            syn(&p, "{value:");
            decode_content(json_get(messages[next], "content"), &content);
            quoted(&p, content.data ? content.data : "", content.len);
            syn(&p, "}<tool_response|>");
            free(id.data);
            free(name.data);
            responses = 1;
        }
        i = next - 1;

        // Close the turn unless the reply continues: a following assistant
        // message, or generation right after the tool results.
        open_turn = 0;
        if (next < count && role_is(messages[next], "assistant", NULL) && (!json_first(calls) || responses)) {
        } else if (responses && next == count) {
            open_turn = 1;
        } else if (json_first(calls) && !responses) {
            syn(&p, "<|tool_response>");
        } else {
            syn(&p, "<turn|>\n");
            previous_assistant = 0;
        }
    }
    free(messages);

    if (!open_turn) syn(&p, "<|turn>model\n");
    else if (thinking) { syn(&p, "<|channel>thought\n"); *in_thought = 1; }

    if (debug_file) fwrite(p.text.data, 1, p.text.len, debug_file);
    tokens_total = tokenize_prompt(tokenizer, &p, token_ids, MAX_CONTEXT);
    if (tokens_total < 0) return "prompt exceeds the context window: context_length_exceeded";
    return NULL;
}

// ----------------------------------------------------------------------------
// Tool calls in the model's output

// Converts one value of the call syntax at *s to JSON, advancing *s.
int call_value(const char **s, Buffer *out) {
    const char *c = json_ws(*s);
    size_t quote = strlen(QUOTE);
    if (!strncmp(c, QUOTE, quote)) {
        const char *end = strstr(c + quote, QUOTE);
        if (!end) return 0;
        append_str(out, "\"");
        append_escaped(out, c + quote, (size_t)(end - c - quote));
        append_str(out, "\"");
        *s = end + quote;
        return 1;
    }
    if (*c == '{' || *c == '[') {
        int object = *c == '{';
        char close = object ? '}' : ']';
        append(out, c, 1);
        c = json_ws(c + 1);
        for (int first = 1; *c != close; first = 0) {
            if (!first) {
                if (*c != ',') return 0;
                c = json_ws(c + 1);
            }
            append_str(out, first ? "" : ",");
            if (object) {
                const char *key = c, *key_end;
                if (!strncmp(c, QUOTE, quote)) {
                    key = c + quote;
                    if (!(key_end = strstr(key, QUOTE))) return 0;
                    c = key_end + quote;
                } else {
                    while (*c && *c != ':') c++;
                    key_end = c;
                    while (key_end > key && strchr(" \t\r\n", key_end[-1])) key_end--;
                }
                c = json_ws(c);
                if (*c++ != ':') return 0;
                append_str(out, "\"");
                append_escaped(out, key, (size_t)(key_end - key));
                append_str(out, "\":");
            }
            if (!call_value(&c, out)) return 0;
            c = json_ws(c);
        }
        append(out, &close, 1);
        *s = c + 1;
        return 1;
    }
    // A bare literal: kept if it is valid JSON, else quoted as a string.
    const char *end = c;
    while (*end && !strchr(",}]", *end)) end++;
    const char *last = end;
    while (last > c && strchr(" \t\r\n", last[-1])) last--;
    size_t len = (size_t)(last - c);
    char *number_end;
    int literal = (len == 4 && !strncmp(c, "true", 4)) || (len == 5 && !strncmp(c, "false", 5)) ||
                  (len == 4 && !strncmp(c, "null", 4));
    if (!literal && len) {
        strtod(c, &number_end);
        literal = number_end == last;
    }
    if (literal) append(out, c, len);
    else { append_str(out, "\""); append_escaped(out, c, len); append_str(out, "\""); }
    *s = end;
    return 1;
}

// Parses call:NAME{key:value,...} into a name and a JSON arguments object.
int parse_call(const char *text, char *name, size_t cap, Buffer *arguments) {
    text = json_ws(text);
    if (strncmp(text, "call:", 5)) return 0;
    text += 5;
    const char *brace = strchr(text, '{');
    if (!brace || (size_t)(brace - text) >= cap) return 0;
    memcpy(name, text, (size_t)(brace - text));
    name[brace - text] = 0;
    buffer_clear(arguments);
    return call_value(&brace, arguments) && !*json_ws(brace);
}

// ----------------------------------------------------------------------------
// Generation

typedef struct {
    Client *client;
    int thought;      // inside a thought channel
    int label;        // reading the channel's label, "thought\n"
    int call;         // inside a tool call
    Buffer text;      // the label, or the call, so far
    int calls;        // tool calls sent
    char names[64][128]; // the tools offered
    int name_count;
} Stream;

void send_delta(Client *client, const char *field, const char *text, size_t len) {
    if (!len || client->failed) return;
    static Buffer delta;
    buffer_clear(&delta);
    append_str(&delta, "{\"");
    append_str(&delta, field);
    append_str(&delta, "\":\"");
    append_escaped(&delta, text, len);
    append_str(&delta, "\"}");
    send_chunk(client, delta.data, NULL, NULL);
}

// llama.cpp's prefill progress, which Prospero shows with an ETA.
void send_progress(Client *client, int total, int cached, int processed, double started) {
    char frame[256];
    int len = snprintf(frame, sizeof(frame),
        "data: {\"id\":\"gemma4\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"" MODEL_ID "\",\"choices\":[],"
        "\"prompt_progress\":{\"total\":%d,\"cache\":%d,\"processed\":%d,\"time_ms\":%d}}\n\n",
        total, cached, processed, (int)((time_seconds() - started) * 1000));
    send_all(client, frame, (size_t)len);
}

// Handles one generated token. Returns nonzero when generation should stop.
int stream_token(Stream *stream, const Tokenizer *tokenizer, int token) {
    // The string delimiter decodes to nothing, but a call needs its text.
    const char *text = token == TOK_QUOTE ? QUOTE : token_text(tokenizer, token);
    switch (token) {
    case TOK_EOS: case TOK_TURN_END: case TOK_TOOL_RESPONSE:
        return 1;
    case TOK_CHANNEL:
        stream->thought = stream->label = 1;
        buffer_clear(&stream->text);
        return 0;
    case TOK_CHANNEL_END:
        if (stream->label) send_delta(stream->client, "reasoning_content", stream->text.data, stream->text.len);
        stream->thought = stream->label = 0;
        return 0;
    case TOK_TOOL_CALL:
        stream->thought = stream->label = 0;
        stream->call = 1;
        buffer_clear(&stream->text);
        return 0;
    case TOK_TOOL_CALL_END: {
        stream->call = 0;
        char name[128];
        static Buffer arguments;
        if (parse_call(stream->text.data ? stream->text.data : "", name, sizeof(name), &arguments)) {
            send_tool_call(stream->client, stream->calls++, name, arguments.data, arguments.len);
        } else { // Not a valid call after all: pass it on as text.
            send_delta(stream->client, "content", "<|tool_call>", 12);
            send_delta(stream->client, "content", stream->text.data, stream->text.len);
            send_delta(stream->client, "content", "<tool_call|>", 12);
        }
        return 0;
    }
    }
    if (stream->call) {
        append_str(&stream->text, text);
    } else if (stream->label) {
        append_str(&stream->text, text);
        char *newline = strchr(stream->text.data, '\n');
        if (newline || stream->text.len > 16) {
            stream->label = 0;
            int named = newline && !strncmp(stream->text.data, "thought\n", 8);
            const char *rest = named ? stream->text.data + 8 : stream->text.data;
            send_delta(stream->client, "reasoning_content", rest, strlen(rest));
        }
    } else {
        send_delta(stream->client, stream->thought ? "reasoning_content" : "content", text, strlen(text));
    }
    return 0;
}

// Whether token may extend a tool call's header, call:NAME{, to one of the
// offered tools, given the header so far.
int header_allows(const Stream *stream, const char *header, size_t header_len, const char *text) {
    size_t text_len = strlen(text);
    if (!text_len) return 0;
    for (int i = 0; i < stream->name_count; i++) {
        char target[160];
        int target_len = snprintf(target, sizeof(target), "call:%s{", stream->names[i]);
        if ((size_t)target_len <= header_len || strncmp(header, target, header_len)) continue;
        size_t left = (size_t)target_len - header_len;
        if (!strncmp(text, target + header_len, text_len < left ? text_len : left)) return 1;
    }
    return 0;
}

// Masks the logits so that the next token is one the protocol allows: a call
// names an offered tool, and is closed before the turn ends.
void constrain(Stream *stream, const Tokenizer *tokenizer, float *values) {
    if (!stream->name_count) values[TOK_TOOL_CALL] = -INFINITY;
    if (!stream->call) return;
    values[TOK_EOS] = values[TOK_TURN_END] = values[TOK_TOOL_RESPONSE] = -INFINITY;
    const char *header = stream->text.data ? stream->text.data : "";
    if (strchr(header, '{')) return; // The name is settled.
    size_t header_len = strlen(header);
    for (int token = 0; token < VOCAB_SIZE; token++)
        if (values[token] > -INFINITY && !header_allows(stream, header, header_len, token_text(tokenizer, token)))
            values[token] = -INFINITY;
}

// ----------------------------------------------------------------------------
// Request handling

void handle_completion(Model *model, InferenceState *state, Client *client, const char *body) {
    int thinking = request_thinking(body), in_thought;
    const char *error = build_prompt(&model->tokenizer, body, thinking, &in_thought);
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
    // Prefill in chunks on the batch grid: sliding-cache writes are
    // contiguous, so a chunk starting off the grid could cross the wrap
    // boundary. After each chunk, report progress, and stop if the client
    // hung up.
    double started = time_seconds();
    send_progress(client, prompt_tokens, common, common, started);
    int done = common, last_index = 0;
    while (done < prompt_tokens && !client->failed) {
        int chunk = BATCH_SIZE - done % BATCH_SIZE;
        if (chunk > prompt_tokens - done) chunk = prompt_tokens - done;
        forward(model, state, token_ids + done, chunk, done);
        done += chunk;
        last_index = chunk - 1;
        send_progress(client, prompt_tokens, common, done, started);
    }
    if (client->failed) {
        // Keep what was prefilled, so a retry of this request resumes it.
        prev_count = done;
        memcpy(prev_ids, token_ids, (size_t)done * sizeof(int));
        fprintf(stderr, "%d prompt tokens (%d cached), client left after %d\n", prompt_tokens, common, done);
        return;
    }
    double prefilled = time_seconds();

    if (debug_file) {
        fprintf(debug_file, "\nprompt ids (common=%d):", common);
        for (int i = 0; i < prompt_tokens; i++) fprintf(debug_file, " %d", token_ids[i]);
        fprintf(debug_file, "\ngenerated:");
    }
    static Stream stream;
    buffer_clear(&stream.text);
    stream.client = client;
    stream.thought = in_thought;
    stream.label = stream.call = stream.calls = stream.name_count = 0;
    for (const char *tool = json_first(json_get(body, "tools")); tool && stream.name_count < 64; tool = json_next(tool))
        if (json_string(json_get(json_get(tool, "function"), "name"), stream.names[stream.name_count], 128) > 0)
            stream.name_count++;

    const char *finish = "length";
    int position = prompt_tokens;
    int end = prompt_tokens + max_new < MAX_CONTEXT ? prompt_tokens + max_new : MAX_CONTEXT;
    for (int first = 1; position < end && !client->failed; first = 0) {
        float *values = logits(model, state, first ? last_index : 0);
        constrain(&stream, &model->tokenizer, values);
        int next = sample(values, VOCAB_SIZE, temperature);
        if (debug_file) fprintf(debug_file, "%s", next == TOK_QUOTE ? QUOTE : token_text(&model->tokenizer, next));
        if (stream_token(&stream, &model->tokenizer, next)) { finish = "stop"; break; }
        forward(model, state, &next, 1, position);
        token_ids[position++] = next;
        keepalive(client);
    }
    if (stream.calls) finish = "tool_calls";
    else if (stream.call) send_delta(client, "content", stream.text.data, stream.text.len); // Cut off.
    prev_count = position;
    memcpy(prev_ids, token_ids, (size_t)position * sizeof(int));
    send_finish(client, finish, prompt_tokens, common, position - prompt_tokens);

    double finished = time_seconds();
    int generated = position - prompt_tokens;
    fprintf(stderr, "%d prompt tokens (%d cached, %.0f tok/s), %d generated (%.1f tok/s), %d calls, %s\n",
            prompt_tokens, common, (prompt_tokens - common) / (prefilled - started + 1e-9),
            generated, generated / (finished - prefilled + 1e-9), stream.calls, finish);
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
    // The path, with any /v1 prefix and query string dropped.
    char *path = strchr(request.data, ' ');
    path = path ? path + 1 : request.data + strlen(request.data);
    if (!strncmp(path, "/v1/", 4)) path += 3;
    path[strcspn(path, " ?")] = 0;
    long content_length = -1;
    for (const char *h = line_end ? line_end + 1 : split; (h = strchr(h, '\n')); )
        if (!strncasecmp(++h, "Content-Length:", 15)) { content_length = strtol(h + 15, NULL, 10); break; }

    if (!strncmp(request.data, "GET ", 4) && (!strcmp(path, "/models") || !strcmp(path, "/models/" MODEL_ID))) {
        // n_ctx is how llama.cpp reports the context size, as Prospero reads it.
        char model[256];
        snprintf(model, sizeof(model),
            "{\"id\":\"" MODEL_ID "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"gemma4.c\","
            "\"meta\":{\"n_ctx\":%d}}", MAX_CONTEXT);
        char listing[300];
        snprintf(listing, sizeof(listing), "{\"object\":\"list\",\"data\":[%s]}", model);
        send_json(&client, path[7] ? model : listing);
        return;
    }
    int matched = !strncmp(request.data, "POST ", 5) && !strcmp(path, "/chat/completions");
    if (!matched) { http_error(&client, 404, "Not Found", "only POST /v1/chat/completions and GET /v1/models are served"); return; }
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
    const char *model_path = MODEL_FILE;
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

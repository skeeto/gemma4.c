// For SetProcessInformation and power throttling, Windows 10 and later.
#define _WIN32_WINNT 0x0A00
#include "win.h"

#include <errno.h>
#include <shellapi.h>
#include <stdlib.h>

void argv_utf8(int *argc, char ***argv) {
    LPWSTR *wide = CommandLineToArgvW(GetCommandLineW(), argc);
    if (!wide) return;
    char **utf8 = malloc(*argc * sizeof(*utf8));
    for (int i = 0; i < *argc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, NULL, 0, NULL, NULL);
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, utf8[i] = malloc(n), n, NULL, NULL);
    }
    LocalFree(wide);
    *argv = utf8;
}

void *mmap(void *addr, size_t length, int protection, int flags, int fd, int64_t offset) {
    (void)addr;
    if (!length || flags != MAP_PRIVATE || !(protection & PROT_READ)) {
        errno = EINVAL;
        return MAP_FAILED;
    }

    HANDLE file = (HANDLE)_get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return MAP_FAILED;
    }

    int writable = protection & PROT_WRITE;
    DWORD page = writable ? PAGE_WRITECOPY : PAGE_READONLY;
    DWORD access = writable ? FILE_MAP_COPY : FILE_MAP_READ;
    HANDLE mapping = CreateFileMappingA(file, NULL, page, 0, 0, NULL);
    if (!mapping) {
        errno = EIO;
        return MAP_FAILED;
    }

    uint64_t position = (uint64_t)offset;
    void *view = MapViewOfFile(mapping, access, (DWORD)(position >> 32), (DWORD)position, length);
    CloseHandle(mapping);
    if (!view) {
        errno = EIO;
        return MAP_FAILED;
    }
    return view;
}

int munmap(void *address, size_t length) {
    (void)length;
    if (UnmapViewOfFile(address)) return 0;
    errno = EIO;
    return -1;
}

// Windows 11 throttles a process it judges to be in the background, such as
// one started without a window or over SSH, onto efficiency cores at low
// clocks a few seconds in, which cut decoding to a third on an i9-12900.
// Inference is the work the user is waiting for, so opt out.
void disable_power_throttling(void) {
    PROCESS_POWER_THROTTLING_STATE state = {
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
}

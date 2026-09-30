/* Test-only application filters, automatically removed with the WFP session. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <initguid.h>
#include <fwpmu.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

static DWORD block_application(HANDLE engine, const wchar_t *path) {
    FWP_BYTE_BLOB *app = NULL;
    DWORD error = FwpmGetAppIdFromFileName0(path, &app);
    if (error) return error;
    FWPM_FILTER_CONDITION0 condition = {0};
    condition.fieldKey = FWPM_CONDITION_ALE_APP_ID;
    condition.matchType = FWP_MATCH_EQUAL;
    condition.conditionValue.type = FWP_BYTE_BLOB_TYPE;
    condition.conditionValue.byteBlob = app;
    const GUID *layers[] = {
        &FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V4, &FWPM_LAYER_ALE_RESOURCE_ASSIGNMENT_V6,
        &FWPM_LAYER_ALE_AUTH_CONNECT_V4, &FWPM_LAYER_ALE_AUTH_CONNECT_V6
    };
    for (size_t i = 0; i < sizeof(layers) / sizeof(layers[0]); ++i) {
        FWPM_FILTER0 filter = {0};
        filter.displayData.name = L"Polyvoice offline consumer";
        filter.layerKey = *layers[i];
        filter.subLayerKey = FWPM_SUBLAYER_UNIVERSAL;
        filter.weight.type = FWP_EMPTY;
        filter.action.type = FWP_ACTION_BLOCK;
        filter.numFilterConditions = 1;
        filter.filterCondition = &condition;
        error = FwpmFilterAdd0(engine, &filter, NULL, NULL);
        if (error) break;
    }
    FwpmFreeMemory0((void **)&app);
    return error;
}

/* The first two arguments are file paths, which cannot contain quotes on Windows.
 * Preserve the remaining command line verbatim, including Python -c quoting. */
static const wchar_t *skip_path(const wchar_t *text) {
    if (*text == L'"') {
        ++text;
        while (*text && *text != L'"') ++text;
        if (*text) ++text;
    } else {
        while (*text && *text != L' ' && *text != L'\t') ++text;
    }
    while (*text == L' ' || *text == L'\t') ++text;
    return text;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) return 2;
    HANDLE engine = NULL;
    FWPM_SESSION0 session = {0};
    session.flags = FWPM_SESSION_FLAG_DYNAMIC;
    DWORD error = FwpmEngineOpen0(NULL, RPC_C_AUTHN_WINNT, NULL, &session, &engine);
    /* Include the base Python executable: Windows venv redirectors may launch it. */
    if (!error) error = block_application(engine, argv[1]);
    if (!error) error = block_application(engine, argv[2]);
    HANDLE job = NULL;
    PROCESS_INFORMATION child = {0};
    wchar_t *command = NULL;
    if (!error) {
        job = CreateJobObjectW(NULL, NULL);
        if (!job) error = GetLastError();
    }
    if (!error) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            error = GetLastError();
    }
    if (!error) {
        command = _wcsdup(skip_path(skip_path(GetCommandLineW())));
        if (!command) error = ERROR_NOT_ENOUGH_MEMORY;
    }
    if (!error) {
        STARTUPINFOW startup = {0};
        startup.cb = sizeof(startup);
        if (!CreateProcessW(argv[2], command, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &startup, &child))
            error = GetLastError();
    }
    DWORD result = 3;
    if (!error) {
        if (!AssignProcessToJobObject(job, child.hProcess)) error = GetLastError();
        if (!error && ResumeThread(child.hThread) == (DWORD)-1) error = GetLastError();
        if (!error && WaitForSingleObject(child.hProcess, INFINITE) != WAIT_OBJECT_0) error = GetLastError();
        if (!error && !GetExitCodeProcess(child.hProcess, &result)) error = GetLastError();
        if (error) TerminateProcess(child.hProcess, 3);
    }
    if (child.hThread) CloseHandle(child.hThread);
    if (child.hProcess) CloseHandle(child.hProcess);
    if (job) CloseHandle(job);
    free(command);
    if (engine) FwpmEngineClose0(engine);
    if (error) {
        fprintf(stderr, "offline Windows filter/child failed: %lu\n", error);
        return 3;
    }
    return (int)result;
}

"""Errors identify the failing layer without silently changing backends."""


class PlatformError(Exception):
    pass


class GraphError(PlatformError):
    pass


class CompileError(PlatformError):
    pass


class ResourceError(CompileError):
    pass


class ExecutionError(PlatformError):
    pass


class BackendUnavailable(ExecutionError):
    pass

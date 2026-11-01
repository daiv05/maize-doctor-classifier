"""Bloqueo exclusivo de archivo portable entre POSIX (fcntl) y Windows (msvcrt)."""

import sys

if sys.platform == "win32":
    import msvcrt

    def acquire_exclusive(handle) -> None:
        """Toma un bloqueo exclusivo no bloqueante sobre el archivo abierto.

        @param handle Archivo abierto cuyo descriptor se bloquea.
        @raises BlockingIOError Si otro proceso ya posee el bloqueo.
        """
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise BlockingIOError(str(error)) from error

    def release(handle) -> None:
        """Libera el bloqueo tomado con `acquire_exclusive`.

        @param handle Archivo abierto que posee el bloqueo.
        """
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def acquire_exclusive(handle) -> None:
        """Toma un bloqueo exclusivo no bloqueante sobre el archivo abierto.

        @param handle Archivo abierto cuyo descriptor se bloquea.
        @raises BlockingIOError Si otro proceso ya posee el bloqueo.
        """
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(handle) -> None:
        """Libera el bloqueo tomado con `acquire_exclusive`.

        @param handle Archivo abierto que posee el bloqueo.
        """
        fcntl.flock(handle, fcntl.LOCK_UN)

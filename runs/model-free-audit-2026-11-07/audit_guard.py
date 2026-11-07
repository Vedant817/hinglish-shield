import sys, os
from pathlib import Path
root = Path(os.environ['HS_AUDIT_ROOT']).resolve()
def audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return
    if path.startswith(('data/seeds/','data/interim/','data/curated/','data/staging/')) and path != 'data/staging/review/vedant_review_nov7_calibration_dev.csv':
        raise PermissionError('Audit forbids mixed/test prompt reads: ' + path)
sys.addaudithook(audit)

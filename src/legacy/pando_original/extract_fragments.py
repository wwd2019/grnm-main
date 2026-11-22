"""Extract fragment files on l40; reject links and paths outside the project."""
from pathlib import Path
import json
import subprocess
import tarfile
import shutil

root = Path('/home/wwd/py_code/mhr/final/new_grn')
destination = (root / 'fragments').resolve()
destination.mkdir(exist_ok=True)
archive = Path('/home/wwd/data/timecourse_fragments.tar.gz')
manifest = []
with subprocess.Popen(['pigz', '-dc', '-p', '4', str(archive)], stdout=subprocess.PIPE) as process:
    with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
        for member in stream:
            print(member.name, member.size, flush=True)
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Invalid archive path: {member.name}')
            if not member.isfile():
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                raise ValueError(f'Unsupported archive member: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.size:
                print('REUSE', target, flush=True)
            else:
                temporary = target.with_name(target.name + '.partial')
                with stream.extractfile(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                temporary.replace(target)
            manifest.append({'member': member.name, 'path': str(target), 'size': member.size})
    if process.wait() != 0:
        raise RuntimeError('pigz decompression failed')
(root / 'fragments_manifest.json').write_text(json.dumps(manifest, indent=2))
print('EXTRACTION_COMPLETE', len(manifest), flush=True)

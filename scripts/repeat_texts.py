import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='../stimuli/',
                        help='Path to the items used in the experiment')
    parser.add_argument('-s', '--scanpaths', type=str, default='../scanpaths/',
                        help='Path to subjects\' scanpaths, divided by item')
    parser.add_argument('-o', '--output', type=str, default='../texts/')
    args = parser.parse_args()
    stimuli, scanpaths, output = Path(args.path), Path(args.scanpaths), Path(args.output)
    if not (stimuli.exists() and scanpaths.exists()):
        raise ValueError('You must specify valid paths for stimuli and scanpaths')
    if not output.exists():
        output.mkdir(parents=True)

    items = [item for item in scanpaths.iterdir() if item.is_dir()]
    for item in items:
        files = [file for file in item.iterdir() if file.is_file()]
        if len(files) > 0:
            item_output = output / item.stem
            item_output.mkdir(exist_ok=True)
            stimulus = stimuli / item.stem
            text = stimulus.read_text().replace('\n', ' ')
            text = text.replace('. ', '.\n')
            for i, _ in enumerate(files):
                with (item_output / f'{i} {stimulus.name}.txt').open('w') as f:
                    f.write(text)

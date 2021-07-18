from ruamel.yaml import YAML
from cerberus import Validator
from colour import CCTF_ENCODINGS

# See https://docs.python-cerberus.org/en/stable/
schema = {
    'LUT': {
        'type': 'dict',
        'schema': {
            'size': {'type': 'integer'},
            'algorithm': {'type': 'string', 'allowed': ['Root-Polynomial', 'Linear Matrix']},
            'gamma': {'type': 'string', 'allowed': CCTF_ENCODINGS.keys()},
            'output_file': {'type': 'string'}
        }
    },
    'files': {
        'type': 'list',
        'valuesrules': {
            'type': 'dict',
            'schema': {
                'type': {'type': 'string'},
                'path': {'type': 'string'},
                'gamma': {'type': 'string', 'allowed': CCTF_ENCODINGS.keys()},
                'start_frame': {'type': 'integer'},
                'fps': {'type': 'number'},
                'samples': {
                    'type': 'dict',
                    'schema': {
                        'batch': {'type': 'integer'},
                        'type': {'type': 'string', 'allowed': ['Colour', 'Grayscale']},
                        'quantity': {'type': 'integer'},
                        'interval': {'type': 'integer'},
                        'csv': {'type': 'string'}
                    }
                }
            }
        }
    }
}


def read_yml_settings(path):
    validator = Validator(schema)
    yaml = YAML(typ='safe')

    try:
        settings = yaml.load(path)
    except FileNotFoundError:
        print(
            f"Error: No settings file found. Create a file called {str(path)} in the current directory.")
    except Exception as exception:
        print("An exception was found.", exception)

    if not validator.validate(settings):
        print("Error: Settings file has incorrect or missing fields:")
        print(validator.errors)
        exit()

    return settings


def extract_file_settings(settings):
    files = settings['files']
    out_files = []

    for file in files:
        out_files.extend(file.values())

    return out_files


def find_num_batches(files):
    batch_numbers = [file['samples']['batch'] for file in files]
    return max(batch_numbers)


def find_curr_batch_files(files, index, patch_type="Colour"):
    source_file = None
    target_file = None

    for file in files:
        is_source = file['type'] == 'Source'
        is_target = file['type'] == 'Target'

        correct_batch_index = file['samples']['batch'] == index
        correct_patch_type = file['samples']['type'] == patch_type

        if is_source and correct_batch_index and correct_patch_type:
            source_file = file

        if is_target and correct_batch_index and correct_patch_type:
            target_file = file

        if source_file and target_file:
            return (source_file, target_file)

    return (None, None)


def find_source_gamma(files):
    return next(iter([file['gamma'] for file in files if file['type']
                      == 'Source']), None)


def find_target_gamma(files):
    return next(iter([file['gamma'] for file in files if file['type']
                      == 'Target']), None)

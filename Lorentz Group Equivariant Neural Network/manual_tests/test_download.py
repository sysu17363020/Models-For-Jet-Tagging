import logging

logging.basicConfig(level=logging.INFO)

from lgn.data.prepare.md17 import download_dataset_md17
download_dataset_md17('/tmp/test', 'md17', 'uracil')

# from lgn.data.prepare.qm9 import download_dataset_qm9
# download_dataset_qm9('/tmp/test', 'qm9')

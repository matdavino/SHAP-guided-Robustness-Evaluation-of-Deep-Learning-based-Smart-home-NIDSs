from os.path import join

_BASE_DATA_PATH = "/nas/data/datasets/"

dataset_config = {
    'cic-iomt': {  # class order
        'path': join(_BASE_DATA_PATH, 'cic-iomt',
                     'cic-iomt_firstDIRfixed_100pkts_7f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'),
    },
    'edge-iiot': {  # class order
        'path': join(_BASE_DATA_PATH, 'edge-iiot',
                    'edge-iiot_firstDIRfixed_100pkts_6f_50p-benign_1p-mt100k_clean_uniform-label.parquet'),
    },
    'ton-iot': {  # class order
        'path': join(_BASE_DATA_PATH, 'ton-iot',
                     'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label.parquet'),
    },

    'ton-iot_6classes': {  # class order
        'path': join(_BASE_DATA_PATH, 'ton-iot',
                     'ton-iot_100pkts_firstDIRfixed_PLfixed_6f_1p-mt1m_10p-mt100k_clean_uniform-label_6classes.parquet'),
    },
}

min_max_config = {
    'PL': (-1,1500), 'IAT': (-1,60000), 'DIR': (0,1), 'WIN': (-1,65535),
}

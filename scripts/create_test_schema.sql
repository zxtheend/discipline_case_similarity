CREATE TABLE IF NOT EXISTS case_similarity_source (
    case_id VARCHAR(64) NOT NULL PRIMARY KEY,
    source_wtxx_bh VARCHAR(64) NULL,
    petition_id VARCHAR(64) NULL,
    location VARCHAR(128) NOT NULL,
    encrypted_reported_persons TEXT NULL,
    encrypted_reporter TEXT NULL,
    encrypted_description LONGBLOB NULL,
    create_time DATETIME NOT NULL,
    KEY idx_case_similarity_source_wtxx (source_wtxx_bh),
    KEY idx_case_similarity_petition (petition_id),
    KEY idx_case_similarity_location (location)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS t_xf_xfj (
    C_BH VARCHAR(64) NOT NULL PRIMARY KEY,
    C_BFYR_XX TEXT NULL,
    C_FYR_XX TEXT NULL,
    C_WTSD_QC VARCHAR(255) NULL,
    DT_CJSJ DATETIME NULL,
    DT_ZHXGSJ DATETIME NULL,
    KEY idx_t_xf_xfj_updated (DT_ZHXGSJ, C_BH)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS t_xf_wtxx (
    C_BH VARCHAR(64) NOT NULL PRIMARY KEY,
    C_XFJ_BH VARCHAR(64) NULL,
    LC_YJMS LONGBLOB NULL,
    DT_CJSJ DATETIME NULL,
    DT_ZHXGSJ DATETIME NULL,
    KEY idx_t_xf_wtxx_xfj (C_XFJ_BH),
    KEY idx_t_xf_wtxx_updated (DT_ZHXGSJ, C_BH)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

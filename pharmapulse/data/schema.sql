-- PharmaPulse star schema (DuckDB). All data is synthetic.

CREATE TABLE brands (
    brand               VARCHAR PRIMARY KEY,
    molecule_class      VARCHAR NOT NULL,
    manufacturer        VARCHAR NOT NULL,
    is_our_brand        BOOLEAN NOT NULL,
    list_price_per_30d  DOUBLE  NOT NULL
);

CREATE TABLE regions (
    region            VARCHAR PRIMARY KEY,
    population_index  DOUBLE NOT NULL
);

CREATE TABLE hcps (
    hcp_id                  VARCHAR PRIMARY KEY,
    specialty               VARCHAR NOT NULL,
    region                  VARCHAR NOT NULL REFERENCES regions(region),
    potential_trx_per_week  DOUBLE  NOT NULL
);

-- Weekly prescription feed at HCP x brand grain (zero rows omitted)
CREATE TABLE weekly_rx (
    hcp_id  VARCHAR NOT NULL REFERENCES hcps(hcp_id),
    week    DATE    NOT NULL,
    brand   VARCHAR NOT NULL REFERENCES brands(brand),
    trx     INTEGER NOT NULL,   -- total prescriptions
    nrx     INTEGER NOT NULL    -- new prescriptions
);

-- Field-force activity for our brand
CREATE TABLE promo (
    hcp_id   VARCHAR NOT NULL REFERENCES hcps(hcp_id),
    week     DATE    NOT NULL,
    brand    VARCHAR NOT NULL REFERENCES brands(brand),
    calls    INTEGER NOT NULL,
    samples  INTEGER NOT NULL
);

-- Payer access: 1 = preferred, 2 = non-preferred, 3 = restricted / prior auth
CREATE TABLE formulary (
    region   VARCHAR NOT NULL,
    brand    VARCHAR NOT NULL,
    year     INTEGER NOT NULL,
    quarter  INTEGER NOT NULL,
    tier     INTEGER NOT NULL
);

CREATE TABLE patients (
    patient_id   VARCHAR PRIMARY KEY,
    age_band     VARCHAR NOT NULL,
    region       VARCHAR NOT NULL,
    index_date   DATE    NOT NULL,
    index_brand  VARCHAR NOT NULL
);

-- Pharmacy claims (fills)
CREATE TABLE claims (
    claim_id     VARCHAR PRIMARY KEY,
    patient_id   VARCHAR NOT NULL REFERENCES patients(patient_id),
    brand        VARCHAR NOT NULL,
    fill_date    DATE    NOT NULL,
    days_supply  INTEGER NOT NULL
);

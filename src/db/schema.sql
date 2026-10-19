-- building / floor / seats / outlets / surveyed_at / surveyor は空欄（NULL）を許す。
-- 選択式の列は「未調査」を有効な値として持つ（条件検索では一致させない）。
CREATE TABLE IF NOT EXISTS spots (
    spot_id         TEXT PRIMARY KEY,
    building        TEXT,
    floor           INTEGER,
    area_name       TEXT NOT NULL,
    seats           INTEGER,
    outlets         INTEGER,
    chair_type      TEXT NOT NULL,
    standing        TEXT NOT NULL,
    quietness       TEXT NOT NULL,
    food            TEXT NOT NULL,
    task_light      TEXT NOT NULL,
    outlet_location TEXT,
    note            TEXT,
    photo           TEXT,
    surveyed_at     DATE,
    surveyor        TEXT,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    spot_id   TEXT NOT NULL REFERENCES spots(spot_id),
    body      TEXT NOT NULL,
    rating    TEXT,
    posted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notices (
    notice_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file     TEXT NOT NULL UNIQUE,
    title           TEXT,
    raw_body        TEXT NOT NULL,
    is_relevant     INTEGER NOT NULL,
    impact_type     TEXT,
    target_building TEXT,
    target_floor    INTEGER,
    target_area     TEXT,
    valid_from      DATE,
    valid_to        DATE,
    confirmed       INTEGER NOT NULL DEFAULT 0
);

-- 教室の利用状況（survey.xlsx の reservations シート）。取込のたびに全件入れ替える。
-- ある日の利用状況を、毎週同じ曜日（weekday。date から決める）の利用状況として使う。
-- spot_id は対応するスポット（空欄なら room とスポットのエリア名で対応づける）。
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date           DATE NOT NULL,
    weekday        TEXT,
    room           TEXT NOT NULL,
    start          TEXT NOT NULL,
    end            TEXT NOT NULL,
    periods        TEXT,
    duplicate      INTEGER NOT NULL DEFAULT 0,
    spot_id        TEXT
);

-- 利用者からの誤り報告（公式データは直接変えない。管理画面で確認する）
CREATE TABLE IF NOT EXISTS error_reports (
    report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    spot_id     TEXT NOT NULL REFERENCES spots(spot_id),
    body        TEXT NOT NULL,
    posted_at   TEXT NOT NULL,
    resolved    INTEGER NOT NULL DEFAULT 0
);

-- キッチンカーの出店情報（管理者が毎月 CSV で取り込む。1行 = 1日・1店・1メニュー）。
-- 同じ月を取り込み直すと、その月の分を置き換える。
CREATE TABLE IF NOT EXISTS kitchen_cars (
    kitchen_car_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date           DATE NOT NULL,
    weekday        TEXT NOT NULL,
    open           TEXT NOT NULL,
    close          TEXT NOT NULL,
    location       TEXT NOT NULL,
    vendor         TEXT NOT NULL,
    item           TEXT NOT NULL,
    price_yen      INTEGER NOT NULL CHECK (price_yen >= 0)
);
CREATE INDEX IF NOT EXISTS kitchen_cars_date ON kitchen_cars (date);

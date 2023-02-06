BEGIN;

CREATE TABLE ratings (
    name text NOT NULL,
    climate text NOT NULL,
    culture text NOT NULL,
    cuisine text NOT NULL,
    adventure_activities text NOT NULL,
    natural_beauty text NOT NULL,
    budget text NOT NULL,
    language text NOT NULL,
    safety text NOT NULL
);

alter table ratings
    owner to postgres;

CREATE TABLE answers (
    name text NOT NULL,
    iceland text NOT NULL,
    maldives text NOT NULL,
    monaco text NOT NULL,
    singapore text NOT NULL,
    egypt text NOT NULL
);
alter table answers
    owner to postgres;
COMMIT;

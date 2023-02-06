FROM postgres
ENV POSTGRES_PASSWORD docker
ENV POSTGRES_DB postgres
COPY create_db.sql /docker-entrypoint-initdb.d/

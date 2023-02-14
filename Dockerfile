FROM postgres
ENV POSTGRES_PASSWORD Os$m4`KQ>y=]>|7Z
ENV POSTGRES_DB postgres
COPY create_db.sql /docker-entrypoint-initdb.d/

#docker build -t eu.gcr.io/wagon-bootcamp-355610/simpledb:prod .
#docker push eu.gcr.io/wagon-bootcamp-355610/simpledb:prod NQRLGaAjR44JM5Zk

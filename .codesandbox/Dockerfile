FROM python:3.9.16
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone
RUN apt update -y && apt install lib32z1 -y
RUN wget  https://sqlite.org/2023/sqlite-tools-linux-x86-3420000.zip && unzip sqlite-tools-linux-x86-3420000.zip
RUN cp sqlite-tools-linux-x86-3420000/* /usr/bin/ 


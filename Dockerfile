FROM python:3.8-slim
RUN mkdir app
WORKDIR "/app/"
COPY . .
RUN pip3 install -r requirements.txt
CMD streamlit run --server.port=${PORT} --server.address "0.0.0.0" --server.headless true
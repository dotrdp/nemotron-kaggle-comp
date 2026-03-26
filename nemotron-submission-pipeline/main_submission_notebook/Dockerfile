
FROM gcr.io/kaggle-gpu-images/python@sha256:9fa0da194fad2241d3f01a80581cbecbd3a258b4d1b695e2cbbbc62a0fd205ac

COPY submission_generator.py /
#COPY requirements.txt /
COPY build.py /

RUN python /build.py

CMD ["python", "/submission_generator.py"]

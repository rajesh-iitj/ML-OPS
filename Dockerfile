FROM python:3.9.17
COPY ./ /digits/
COPY ./requirements.txt /digits/
RUN pip3 install -r /digits/requirements.txt
WORKDIR /digits
CMD ["pytest"]
#CMD ["python", "plot_digits_classification.py"]
#ENV FLASK_APP=api/app
#CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
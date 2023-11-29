# Use the public ECR image as the base
FROM agrigorev/zoomcamp-bees-wasps:v2


RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# set the name
ENV MODEL_NAME=bees-wasps-v2.tflite

# copy doc
COPY lambda_function.py .

# run function
CMD ["lambda_function.lambda_handler"]
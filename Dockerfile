FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum install -y gcc gcc-c++ make

# Install Python packages
RUN pip install pandas scipy pywavelets boto3 requests -t .

# Copy function code
COPY app.py .
COPY emg_processor.py .

# Set entrypoint
CMD ["app.lambda_handler"]

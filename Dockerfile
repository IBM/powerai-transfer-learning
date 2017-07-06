FROM jarvice/ubuntu-ibm-mldl-ppc64le

# uprev to force rebuild even if cached
ENV APP_POWERAI_VERSION 1

ADD https://raw.githubusercontent.com/nimbix/notebook-common/master/install-ubuntu.sh /tmp/install-ubuntu.sh
RUN bash /tmp/install-ubuntu.sh && rm -f /tmp/install-ubuntu.sh

COPY NAE/help.html /etc/NAE/help.html

COPY NAE/AppDef.json /etc/NAE/AppDef.json
COPY NAE/screenshot.png /etc/NAE/screenshot.png
RUN curl --fail -X POST -d @/etc/NAE/AppDef.json https://api.jarvice.com/jarvice/validate

COPY samples /usr/local/samples
COPY scripts/sample_notebook.sh /usr/local/scripts/sample_notebook.sh

#add Jupyter
RUN pip install --upgrade pip
RUN pip install notebook pyyaml
RUN pip install jupyter
RUN pip install ijson
RUN pip install pandas
RUN pip install python-resize-image

RUN pip install pandas_datareader
RUN pip install httplib2
RUN apt-get install -y build-essential
RUN apt-get install -y libssl-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y python-dev
RUN apt-get install -y python-matplotlib
RUN apt-get install -y python-lxml
RUN apt-get install -y openssh-server
RUN pip install cython
RUN apt-get install -y python-scipy
RUN pip install scikit-learn
RUN apt-get install -y libxml2-dev libxmlsec1-dev


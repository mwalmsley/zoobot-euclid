FROM gitlab.euclid-sgs.uk:4567/st-tools/ct_xodeen_builder/dockeen

RUN ["/bin/bash", "-c", "echo hello"]

RUN ["which", "python"]

# added by volume mount
# for production use, would need to COPY (host) (target)
WORKDIR /home/user/zoobot-euclid

# added in .bashrc
RUN eden.3.0

# RUN pip3 install --user -e /home/user/zoobot-euclid

# CMD ["python", "euclid_morphology/inference.py"]
#!/usr/bin/env bash

gunicorn wsgi:app --bind 0.0.0.0:80 --log-level=debug --workers=2
#!/usr/bin/bash
copy_create() {
    cp steps/step$1.py steps/step$2.py
    code steps/step$2.py
}
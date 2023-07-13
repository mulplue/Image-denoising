#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
from PIL import Image


def write_params(log_path, parser, description=None):
    opt = parser.parse_args()
    options = parser._optionals._actions
    with open(log_path+'params.md', 'w+') as file:
        file.write('# Params\n')
        file.write('********************************\n')
        file.write('Time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        if description is not None:
            file.write('**Description**: '+description+'\n')
        
        file.write('| Param | Value | Description |\n')
        file.write('| ----- | ----- | ----------- |\n')
        for i in range(len(parser._optionals._actions)):
            option = options[i]
            if option.dest != 'help':
                file.write('|**'+ option.dest+'**|'+str(opt.__dict__[option.dest])+'|'+option.help+'|\n')
        file.write('********************************\n\n')
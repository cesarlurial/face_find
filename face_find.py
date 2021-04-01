# Original https://github.com/sipeed/MaixPy_scripts/blob/master/machine_vision/demo_find_face.py
# for MAiX BiT

import sensor
import image
import lcd
import time
import KPU as kpu

clock = time.clock()
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(0)
sensor.set_vflip(1)
sensor.run(1)
task = kpu.load(0x300000) # you need put model(face.kfpkg) in flash at address 0x300000
# task = kpu.load("/sd/face.kmodel")
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
a = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)
while(True):
    clock.tick()
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    if code:
        for i in code:
            print(i)
            a = img.draw_rectangle(i.rect())
    fps =clock.fps()
    a = img.draw_string(2,2, ("%2.1ffps" %(fps)), color=(0,128,0), scale=2)
    a = lcd.display(img)
a = kpu.deinit(task)

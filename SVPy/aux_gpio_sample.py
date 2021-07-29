from gpiozero import LED
from time import sleep

led_green = LED(18)
led_white = LED(24)
led_red = LED(8)
led_yellow = LED(12)
led_blue = LED(20)

leds = [led_green, led_white, led_red, led_yellow, led_blue]

while True:
    for led in leds:
        led.on()
        sleep(1)
    for led in leds:
        led.off()
        sleep(0.5)
    for led in leds:
        led.blink(0.21, 0.09)
        sleep(1)
    for led in leds:
        led.off()
        sleep(0.5)
    for led in leds:
        for l in leds:
            l.off()
        led.on()
        sleep(1)
    for led in leds:
        led.off()
    sleep(0.5)
    for led in leds:
        for l in leds:
            l.off()
        led.blink(0.21, 0.09)
        sleep(1)
    for led in leds:
        led.off()
    sleep(0.5)
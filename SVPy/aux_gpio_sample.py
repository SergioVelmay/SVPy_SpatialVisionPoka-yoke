from gpiozero import LED, Buzzer
from time import sleep

led_green = LED(21)
led_white = LED(16)
led_red = LED(12)
led_yellow = LED(1)
led_blue = LED(8)

leds = [led_green, led_white, led_red, led_yellow, led_blue]

buzzer = Buzzer(24)

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

    buzzer.on()
    sleep(0.200)
    buzzer.off()

    sleep(2)

    buzzer.on()
    sleep(0.200)
    buzzer.off()

    sleep(2)

    buzzer.on()
    sleep(0.500)
    buzzer.off()

    sleep(2)

    buzzer.on()
    sleep(0.500)
    buzzer.off()

    sleep(2)

    buzzer.on()
    sleep(0.075)
    buzzer.off()
    sleep(0.050)
    buzzer.on()
    sleep(0.075)
    buzzer.off()

    sleep(1)

    buzzer.on()
    sleep(0.075)
    buzzer.off()
    sleep(0.050)
    buzzer.on()
    sleep(0.075)
    buzzer.off()

    sleep(1)

    buzzer.on()
    sleep(0.050)
    buzzer.off()
    sleep(0.025)
    buzzer.on()
    sleep(0.050)
    buzzer.off()
    sleep(0.025)
    buzzer.on()
    sleep(0.050)
    buzzer.off()

    sleep(1)

    buzzer.on()
    sleep(0.050)
    buzzer.off()
    sleep(0.025)
    buzzer.on()
    sleep(0.050)
    buzzer.off()
    sleep(0.025)
    buzzer.on()
    sleep(0.050)
    buzzer.off()

    sleep(1)

    buzzer.blink(0.15, 0.05)
    sleep(3)
    buzzer.off()

    sleep(1)
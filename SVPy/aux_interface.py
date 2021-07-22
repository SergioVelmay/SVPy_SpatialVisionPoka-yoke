from tkinter import Tk, Label, N
 
class Window():
 
    def __init__(self):
 
        self.root = Tk()
        self.root.wm_title('SVPy | Spatial Vision Poka-yoke')
        self.root.geometry('960x540')
        self.root.resizable(False, False)
 
        from aux_constants import colors_hex, welcome

        self.streaming = Label(self.root, borderwidth = 0, bg='white')

        self.instruction = Label(self.root, borderwidth = 0, bg='black', fg='white')
        self.instruction.config(font=('Tahoma', 14, 'bold'), wraplength=300, text=welcome)

        self.validation = Label(self.root, borderwidth = 0, bg=colors_hex['aux'])
        self.assembly = Label(self.root, borderwidth = 0, bg=colors_hex['aux'])

        self.currently = Label(self.root, borderwidth = 0, bg='black', fg='white')
        self.currently.config(font=('Tahoma', 14), wraplength=300)

        self.detections = Label(self.root, borderwidth = 0, bg='black', fg='white')
        self.detections.config(font=('Tahoma', 14), wraplength=300, anchor=N)

        self.inference = Label(self.root, borderwidth = 0, bg='black', fg='white')
        self.inference.config(font=('Tahoma', 14), wraplength=150)

        self.total = Label(self.root, borderwidth = 0, bg='black', fg='white')
        self.total.config(font=('Tahoma', 14), wraplength=150)

        self.streaming.place(x=0, y=0, width=640, height=480)
        self.instruction.place(x=640, y=0, width=320, height=80)
        self.validation.place(x=640, y=80, width=160, height=160)
        self.assembly.place(x=800, y=80, width=160, height=160)
        self.currently.place(x=640, y=240, width=320, height=60)
        self.detections.place(x=640, y=300, width=320, height=120)
        self.inference.place(x=640, y=420, width=160, height=60)
        self.total.place(x=800, y=420, width=160, height=60)

        from aux_images import alpha_images

        self.bar_images = []
        self.bar_progress = []

        for n in range(8):
            self.bar_images.append(Label(self.root, borderwidth=0))
            self.bar_images[n].config(bg=colors_hex['aux'], image=alpha_images[n])
            self.bar_images[n].place(x=60*(2*n), y=480, width=60, height=60)
            self.bar_progress.append(Label(self.root, borderwidth=0))
            self.bar_progress[n].config(bg=colors_hex['aux'])
            self.bar_progress[n].place(x=60*(2*n+1), y=480, width=60, height=60)
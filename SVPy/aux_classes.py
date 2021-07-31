class Classification:

    def __init__(self, label, score):
        self.Label = label
        self.Score = score

    def __str__(self):
        words = str.upper(self.Label).split('.')
        if (self.Label.startswith('Step') or self.Label.startswith('Part')):
            if len(words[0]) > 4:
                words[0] = words[0][:4] + ' #' + words[0][4:]
        return ' - '.join(tuple(words)) + ' ({:.1f}'.format(self.Score)[0:6] + '%)'

class Boundary:

    def __init__(self, x, y, w, h):
        self.Left = x
        self.Top = y
        self.Width = w
        self.Height = h

    def __str__(self):
        return 'x:{:.2f} y:{:.2f} w:{:.2f} h:{:.2f}'.format(
            self.Left, self.Top, self.Width, self.Height)

class Detection(Classification):

    def __init__(self, label, score, x, y, w, h):
        Classification.__init__(self, label, score)
        self.Box = Boundary(x, y, w, h)

class Validation(Detection):
    def __init__(self, label, score, x, y, w, h, color, thickness, text):
        self.Detection = Detection(label, score, x, y, w, h)
        self.Color = color
        self.Thickness = thickness
        self.Text = text
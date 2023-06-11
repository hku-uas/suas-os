class FoundEntry:
    def __init__(self, time, item_type, shape, letter, bg_colour, fg_colour, confidence, x, y, img):
        self.time = time
        self.item_type = item_type
        self.shape = shape
        self.letter = letter
        self.bg_colour = bg_colour
        self.fg_colour = fg_colour
        self.confidence = confidence
        self.x = x
        self.y = y
        self.img = img

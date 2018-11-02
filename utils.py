import os

_writeup_dir = './writeup_dir'

'''
save_fig_for_writeup: Function to save a matplot lib figure as image file

    @fig : matplot lib fig
    @ fname : base file name of image file
'''
def save_fig_for_writeup(fig, fname):
    assert(fig is not None)
    file_path = os.path.join(_writeup_dir, fname)
    fig.savefig(file_path)


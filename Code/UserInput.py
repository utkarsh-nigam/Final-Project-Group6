import tkinter as tk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os,sys

array=np.zeros((200,200),int)
new_array=[]
class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=200, height=200, bg = "white", cursor="circle",insertwidth=5)
        self.canvas.pack(side="top", fill="both", expand=True)
        #self.button_print = tk.Button(self, text = "Display points", command = self.print_points)
        #self.button_print.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_submit = tk.Button(self, text="Submit", command=self.submit)
        self.button_submit.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

    def clear_all(self):
        self.points_recorded = []
        self.canvas.delete("all")

    def submit(self):
        global new_array
        global app
        new_array=self.points_recorded
        tk.Tk.destroy(self)
        sys.exit()

    def print_points(self):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()
        self.canvas.create_oval(self.points_recorded, fill = "black")
        self.points_recorded[:] = []

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_oval(self.previous_x, self.previous_y,
                                self.x, self.y,fill="black",width=5)
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)
        self.points_recorded.append(self.x)
        self.previous_x = self.x
        self.previous_y = self.y

if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()

array=np.full((200, 200),255)
my_array=new_array
#print(len(my_array))
x_array=[]
y_array=[]
for i in range(0,int((len(my_array))/2)):
    cord_1,cord_2=my_array[(2*i)+1],my_array[2*i],
    #print(cord_1,cord_2)
    #x_array.append(-(cord_2-200))
    #y_array.append(-(cord_1-200))
    x_array.append(cord_2)
    y_array.append(cord_1)
    for k in range(0,10):
        for j in range(0, 10):
            array[cord_1+5-k,cord_2+5-j]=0

import matplotlib.pyplot as plt
plt.imshow(array,cmap="gray")
plt.scatter(x_array,y_array,color="w")
plt.show()
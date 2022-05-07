# Linear solver
import win32api


def my_linfit(x,y):
    a = 0
    b = 0
    sumy = 0
    sumx = 0
    sumxy = 0
    sumx2 = 0
    i = 0
    # len(y) = len(x)
    while i < len(y):
        sumy = sumy + y[i]
        sumx = sumx + x[i]
        sumxy = sumxy + (x[i]*y[i])
        sumx2 = sumx2 + (x[i]*x[i])
        i = i + 1

    i = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    while i < len(y):
        t1 = t1 + ((x[i]*x[i]*sumx*sumxy)/(sumx2*sumx2))
        t2 = t2+ ((y[i]*x[i]*sumx)/sumx2)
        t3 = t3 + ((x[i]*sumxy)/sumx2)
        t4 = t4 + ((x[i]*sumx)/sumx2)**2
        t5 = t5 + ((x[i]*sumx)/sumx2)
        i = i + 1

    b = (sumy+t1-t2-t3)/(t4-2*t5+len(y))
    a = (sumxy-b*sumx)/sumx2
    return a , b

# Main
def main():
    import matplotlib.pyplot as plt
    import numpy as np
    xp1 = np.arange(0,10,0.1)
    xp2 = [0] * 100

    plt.plot(xp1,xp2) # These are used only for scaling the figure
    plt.plot(xp2,xp1) # These can be considered as x- and y-axises of the figure.

    plt.title("Give N points by using left button of mouse \n"
              "Stop giving points by clicking right button of mouse.")
    collector = plt.ginput(-1,0,True,1,2,3)
    plt.close()
    x = []
    y = []
    for i in collector:
        x.append(i[0])
        y.append(i[1])
    a, b = my_linfit(x, y)
    plt.plot(x, y, 'kx')
    xp = np.arange(0,10,0.1)
    plt.plot(xp, a*xp+b, 'r')
    plt.title(" The linear model generated from the points you gave with magic.")

    plt.plot(xp1,xp2) # These are used only for scaling the figure
    plt.plot(xp2,xp1) # to instantly comparable to the first figure.
                      # They can be considered as x- and y-axises of the figure.

    print(f"My fit : a={a} and b={b}")
    plt.show()
main()

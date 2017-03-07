import pygame

pygame.init()

window = pygame.display.set_mode((800,600))

pygame.display.set_caption("Animation")

white = (255,255,255)

clock = pygame.time.Clock()

# Mario sprites
#fullname = os.path.join('Users/JayBaek/Desktop', "m1.png")
#m1 = pygame.image.load(fullname)
w1 = pygame.image.load("/Users/JayBaek/Desktop/w1.png")
w2 = pygame.image.load("/Users/JayBaek/Desktop/w2.png")
w3 = pygame.image.load("/Users/JayBaek/Desktop/w3.png")
w4 = pygame.image.load("/Users/JayBaek/Desktop/w4.png")
w5 = pygame.image.load("/Users/JayBaek/Desktop/w5.png")



marioCurrentImage = 1

gameLoop=True

while gameLoop: 
    for event in pygame.event.get(): 
        if (event.type==pygame.QUIT): 
            gameLoop=False 
        window.fill(white) 
        if (marioCurrentImage==1): 
            window.blit(w1, (10,10)) 
        if (marioCurrentImage==2): 
            window.blit(w3, (10,20))
        if (marioCurrentImage==3): 
            window.blit(w4, (10,30))
        if (marioCurrentImage==4): 
            window.blit(w5, (10,10))
        if (marioCurrentImage==5): 
            marioCurrentImage=1 
        else: marioCurrentImage+=1; 
    pygame.display.flip() 
    clock.tick(6)

pygame.quit()

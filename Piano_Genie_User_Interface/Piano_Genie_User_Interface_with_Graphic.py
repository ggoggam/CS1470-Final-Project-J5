import numpy as np
import pygame
import simpleaudio as sa

def decoder(computer_key):
    #### This is a RANDOM DECODER for testing purpose only!
    #### It returns ranome note numbers
    if   computer_key == 1: note_number = np.random.randint(21,  35)
    elif computer_key == 2: note_number = np.random.randint(35,  50) 
    elif computer_key == 3: note_number = np.random.randint(50,  55)
    elif computer_key == 4: note_number = np.random.randint(55,  60)
    elif computer_key == 5: note_number = np.random.randint(60,  65)
    elif computer_key == 6: note_number = np.random.randint(65,  70)
    elif computer_key == 7: note_number = np.random.randint(70,  95)
    elif computer_key == 8: note_number = np.random.randint(95, 108)
    return note_number

def draw_line(computer_key, note_number):
    if computer_key != 999:       
        if computer_key < 5: color = blue
        elif computer_key >= 5: color = blue 
        line_x0 = position_dict[note_number][0]
        line_y0 = position_dict[note_number][1]
        pygame.draw.line(display_surface, color, (line_x0, line_y0), (line_x0, line_y0+100), 10)
    else: pass

position_dict = {21:( 21.000,120),22:( 31.125, 80),23:( 41.250,120),24:( 61.500,120),25:( 71.625, 80),
                 26:( 81.750,120),27:( 91.875, 80),28:(102.000,120),29:(122.250,120),30:(132.375, 80),
                 31:(142.500,120),32:(152.625, 80),33:(162.750,120),34:(172.875, 80),35:(183.000,120),
                 36:(203.250,120),37:(213.375, 80),38:(223.500,120),39:(233.625, 80),40:(243.750,120),
                 41:(264.000,120),42:(274.125, 80),43:(284.250,120),44:(294.375, 80),45:(304.500,120),
                 46:(314.625, 80),47:(324.750,120),48:(345.000,120),49:(355.125, 80),50:(365.250,120),
                 51:(375.375, 80),52:(385.500,120),53:(405.750,120),54:(415.875, 80),55:(426.000,120),
                 56:(436.125, 80),57:(446.250,120),58:(456.375, 80),59:(466.500,120),60:(486.750,120),
                 61:(496.875, 80),62:(507.000,120),63:(517.125, 80),64:(527.250,120),65:(547.500,120),
                 66:(557.625, 80),67:(567.750,120),68:(577.875, 80),69:(588.000,120),70:(598.125, 80),
                 71:(608.250,120),72:(628.500,120),73:(638.625, 80),74:(648.750,120),75:(658.875, 80),
                 76:(669.000,120),77:(689.250,120),78:(699.375, 80),79:(709.500,120),80:(719.625, 80),
                 81:(729.750,120),82:(739.875, 80),83:(750.000,120),84:(770.250,120),85:(780.375, 80),
                 86:(790.500,120),87:(800.625, 80),88:(810.750,120),89:(831.000,120),90:(841.125, 80),
                 91:(851.250,120),92:(861.375, 80),93:(871.500,120),94:(881.625, 80),95:(891.750,120),
                 96:(912.000,120),97:(922.125, 80),98:(932.250,120),99:(942.375, 80),100:(952.500,120),
                 101:(972.750,120),102:(982.875, 80),103:(993.000,120),104:(1003.125, 80),105:(1013.250,120),
                 106:(1023.375, 80),107:(1033.500,120),108:(1053.750,120)}



#=====================================
#                                     
#  Piano Jeanie using Random Decoder   
#
#=====================================

pygame.init()

display_surface = pygame.display.set_mode((1080,300))
image = pygame.image.load("keyboard_image.png").convert()
white = (255, 255, 255)
red   = (255,   0,   0)
blue  = (  0,   0, 255)
display_surface.fill(white)
display_surface.blit(image,(10,50))
pygame.display.update()

my_piano = np.load("my_piano.npy", allow_pickle=True)

print()
print("WARNING: You should press ESCAPE after playing!!!") # Erase this line.
print("WARNING: You should press ESCAPE after playing!!!") # Erase this line.
print()
print("=================================================")
print("          1  2  3  4  5  6  7  8                 ")
print("=================================================")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if  event.key == pygame.K_ESCAPE:
                running = False 
            elif event.key == pygame.K_1:
                computer_key = 1
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_1 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_2:
                computer_key = 2
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_2 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_3:
                computer_key = 3
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_3 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_4:
                computer_key = 4
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_4 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_5:
                computer_key = 5
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_5 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_6:
                computer_key = 6
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_6 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_7:
                computer_key = 7
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_7 = [computer_key, note_number]
                my_piano[note_number - 21].play()
            elif event.key == pygame.K_8:
                computer_key = 8
                note_number = decoder(computer_key) # Note number from RNN DECODER
                C_8 = [computer_key, note_number]
                my_piano[note_number - 21].play()

        display_surface.fill(white)
        display_surface.blit(image,(10,50))
        if  event.type == pygame.KEYDOWN:
            if pygame.key.get_pressed()[pygame.K_1]: draw_line(C_1[0],C_1[1])
            if pygame.key.get_pressed()[pygame.K_2]: draw_line(C_2[0],C_2[1])
            if pygame.key.get_pressed()[pygame.K_3]: draw_line(C_3[0],C_3[1])
            if pygame.key.get_pressed()[pygame.K_4]: draw_line(C_4[0],C_4[1])
            if pygame.key.get_pressed()[pygame.K_5]: draw_line(C_5[0],C_5[1])
            if pygame.key.get_pressed()[pygame.K_6]: draw_line(C_6[0],C_6[1])
            if pygame.key.get_pressed()[pygame.K_7]: draw_line(C_7[0],C_7[1])
            if pygame.key.get_pressed()[pygame.K_8]: draw_line(C_8[0],C_8[1])
            pygame.display.update()

pygame.quit()


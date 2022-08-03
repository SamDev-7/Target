import pygame
import math
import random
import neat
import os
import pickle

# Init pygame
pygame.init()

# Set the window size. You may change this to fit your screen.
WIN_WIDTH = 800
WIN_HEIGHT = 800

# Creating and loading resources
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT)) # Create the display
MISSLE_IMG = pygame.transform.scale(pygame.image.load('assets/tile_0012.png').convert_alpha(), (25, 25)) # Missle image
FLAG_IMG = pygame.transform.scale(pygame.image.load('assets/tile_0096.png').convert_alpha(), (25, 25)) # Flag image
FONT = pygame.font.SysFont("comicsans", 20) # Comic Sans font
pygame.display.set_caption("Target Seeking") # Set the window title

class Missle:
    """
    A missle object in game. Acts as the player

    Moves forward at a constant speed
    Can rotate based on input (turn left or right)
    """

    def __init__ (self):
        
        self.speed = 10 # Movement speed in pixels (px) per tick
        self.rotation = random.randint(1, 360) # Initial rotation
        self.rot_speed = 10 # Rotation speed in degrees per tick

        self.vel_vector = pygame.math.Vector2(1, 0) # Using a vector for ease of movement when rotated
        self.vel_vector.rotate_ip(-self.rotation) # Sets up the vector based on the initial rotation

        self.original_image = MISSLE_IMG # Save the original image for rotating
        self.image = self.original_image # Set the rotated image to the original image

        self.rect = self.image.get_rect() # Get the rect of the image
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Random color

        self.rect.x = random.randint(50, WIN_WIDTH-self.rect.width-50) # Random x position (with a margin of 50px)
        self.rect.y = random.randint(50, WIN_HEIGHT-self.rect.height-50) # Random y position (with a margin of 50px)

        self.age = 600 # Initial age of the missle. Counts down and missle dies at 0. Used to remove missles that are not reaching targets

        self.flag = Flag(self.color) # Generates a flag for the missle. Acts like the target

        self.hits = 0 # Stores the number of targets reached
    
    def move(self, left:bool = False, right:bool =False):
        """
        Moves the missle forward. Controls rotation

        Parameters:
            left (bool): Whether the missle should rotate left
            right (bool): Whether the missle should rotate right
        """

        if left:
            # Rotates the missle to the left
            self.rotation = (self.rotation + self.rot_speed) % 360
            self.vel_vector.rotate_ip(-self.rot_speed)
        if right:
            # Rotates the missle to the right
            self.rotation = (self.rotation - self.rot_speed) % 360
            self.vel_vector.rotate_ip(self.rot_speed)

        # Moves the missle forward based on the rotation
        self.rect.center += self.vel_vector*self.speed

        # Recalculate the rect of the image
        self.rect = self.image.get_rect(center=self.rect.center)
        
    def draw(self):
        """
        Draws the missle and flag to the screen
        """

        # Draw the missle
        self.image = pygame.transform.rotate(self.original_image, self.rotation-90)
        WIN.blit(self.image, self.rect) 

        # Draw debug info
        pygame.draw.rect(WIN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
        pygame.draw.line(WIN, self.color, self.rect.center, self.flag.rect.center, 2)

        # Draw the flag
        self.flag.draw()

class Flag:
    """
    A flag object in game. Acts as the target

    Stationary at a random location
    Missle scores when it hits its designated flag
    """

    def __init__ (self, color = (0,0,0)):

        self.image = FLAG_IMG # Set the image for the flag

        self.rect = self.image.get_rect() # Get the rect of the image

        self.rect.x = random.randint(50, WIN_WIDTH-self.rect.width-50) # Random x position (with a margin of 50px)
        self.rect.y = random.randint(50, WIN_HEIGHT-self.rect.height-50) # Random y position (with a margin of 50px)
        self.color = color # Sets the color

    def draw(self):
        """
        Draws the flag to the screen
        """

        # Draw the flag
        WIN.blit(self.image, self.rect)

        # Draw debug info
        pygame.draw.rect(WIN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)

# Used to debug
drawing = False # Whether to update the screen. Faster when set to False.
superspeed = True # Weather to run the program at max speed. Faster when set to True.

def eval_genomes(genomes, config, testing = False):
    """
    Main function for running the game
    """
    global drawing, superspeed

    # If we are testing, not training
    if testing:
        drawing = True
        superspeed = False
    
    # Init some lists
    nets = [] # List of neural networks
    missles = [] # List of missles
    ge = [] # List of genomes

    # Create a net and missle for each genome
    for genome_id, genome in genomes:
        genome.fitness = 0 # Set the init fitness to 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        missles.append(Missle())
        ge.append(genome)

    run = True # Whether the game is running
    clock = pygame.time.Clock() # Used to control the framerate

    hits = [] # Stores the hits of each missle. Used to calculate average hits.

    def calc_angle(x1:int, y1:int, x2:int, y2:int) -> int:
        """
        Calculate the angle from point 1 to point 2
        0 is straight right, 90 is straight up

        Parameters:
            x1 (int): X coordinate of point 1
            y1 (int): Y coordinate of point 1
            x2 (int): X coordinate of point 2
            y2 (int): Y coordinate of point 2
        Returns:
            angle (int): An angle, in degrees
        """
        # To avoid calculation errors.
        if x1 == x2:
            if y1 > y2:
                angle = 270
            else:
                angle = 90
        if y1 == y2:
            if x1 > x2:
                angle = 180
            else:
                angle = 0

        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        angle = -angle % 360
        return angle

    def calc_diff(angle_1:int, angle_2:int) -> int:
        """
        Calculate the difference between two angles (With negatives)

        Parameters:
            angle_1 (int): First angle, in degrees
            angle_2 (int): Second angle, in degrees
        Returns:
            diff (int): The difference between the angles, in degrees
        """
        diff = angle_1 - angle_2

        return diff

    # Running the game
    while run and len(missles) > 0:

        # Set the framerate limit if we are not trying to run as fast a possible
        if not superspeed:
            clock.tick(30)

        for event in pygame.event.get():

            # To quit the game
            if event.type == pygame.QUIT: 
                run = False
                pygame.quit()
                quit()
                break
            
            # To toggle drawing/superspeed (h and s keys)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    drawing = not drawing
                    WIN.fill((0, 0, 0)) # Clear the screen
                    pygame.display.update()
                if event.key == pygame.K_s:
                    superspeed = not superspeed

        def remove(missle:Missle):
            """
            Removes a missle and its flag from the game

            Parameters:
                missle (Missle): The missle to remove
            """

            hits.append(missle.hits) # Add the hits of the missle to the list of hits
            nets.pop(missles.index(missle)) # Remove the missle from the list of nets
            ge.pop(missles.index(missle)) # Remove the missle from the list of genomes
            missles.pop(missles.index(missle)) # Remove the missle from the list of missles, deleting it

        # Runs the game for each missle
        for x, missle in enumerate(missles):
            missle.age -= 1 # Decrease the age so it counts down
            ge[x].fitness -= 2 # Decrease the fitness to discourage not doing anything
            
            # Input: Diff in rotation
            output = nets[x].activate((calc_diff(calc_angle(missle.rect.center[0], missle.rect.center[1], missle.flag.rect.center[0], missle.flag.rect.center[1]),missle.rotation),))
            # Outputs: Rotate left, Rotate right
            missle.move(output[0] > 0.5, output[1] > 0.5)

            # Decreases the fitness if it is rotated. This is to encourage straight paths
            if output[0] > 0.5 or output[1] > 0.5:
                ge[x].fitness -= 1

            # Detects if a missle has hit its target/flag
            if missle.rect.colliderect(missle.flag.rect):
                ge[x].fitness += 1000 # 1000 fitness points reward

                missle.age = 600 # Reset the age
                missle.flag = Flag(missle.color) # Create a new flag
                missle.hits += 1 # Increase the hits

            # If the missle hits the walls
            if missle.rect.left < 0 or missle.rect.right > WIN_WIDTH:
                ge[x].fitness -= 10 # Decreases fitness, to discourage hitting the walls

                # Bounce the missle back
                missle.vel_vector.x *= -1
                missle.rotation = (180-missle.rotation) % 360
            if missle.rect.top < 0 or missle.rect.bottom > WIN_HEIGHT:
                ge[x].fitness -= 10 # Decreases fitness, to discourage hitting the walls

                # Bounce the missle back
                missle.vel_vector.y *= -1
                missle.rotation = (-missle.rotation) % 360

            # This is to insure the missle does not get stuck out of bounds
            if missle.rect.left < 0:
                missle.rect.x = 0
            if missle.rect.right > WIN_WIDTH:
                missle.rect.x = WIN_WIDTH-missle.rect.width
            if missle.rect.top < 0:
                missle.rect.y = 0
            if missle.rect.bottom > WIN_HEIGHT:
                missle.rect.y = WIN_HEIGHT-missle.rect.height

            # Remove the missle if it reaches the desired fitness, so the generation does not run forever
            if ge[x].fitness > 1000000:
                remove(missle)
                continue
            
            # Removes the missle if it has not reached a target/flag for a while
            # The missle will never be removed when testing the best genome
            if missle.age < 0 and not testing:
                remove(missle)

        def draw():
            """
            Draws the game to the screen
            """

            WIN.fill((0, 0, 0)) # Clear the screen

            # Draws each missle
            for missle in missles:
                missle.draw() # Draw the missle

                score = FONT.render(str(ge[missles.index(missle)].fitness), True, missle.color) # Renders the fitness next to the missle

                # DEBUG, uncomment to use
                #best = FONT.render(str(calc_angle(missle.rect.center[0], missle.rect.center[1], missle.flag.rect.center[0], missle.flag.rect.center[1])), True, missle.color)
                #current = FONT.render(str(missle.rotation), True, missle.color)
                #diff = FONT.render(str(calc_diff(calc_angle(missle.rect.center[0], missle.rect.center[1], missle.flag.rect.center[0], missle.flag.rect.center[1]),missle.rotation)), True, missle.color)
                
                WIN.blit(score, (missle.rect.topright[0]+5, missle.rect.topright[1])) # Draws the score

                # DEBUG, uncomment to use
                #WIN.blit(best, (missle.rect.topright[0]+5, missle.rect.topright[1]+15))
                #WIN.blit(current, (missle.rect.topright[0]+5, missle.rect.topright[1]+30))
                #WIN.blit(diff, (missle.rect.topright[0]+5, missle.rect.topright[1]+45))

            pygame.display.update() # Updates the screen
        
        # Check if drawing is enabled
        if drawing:
            draw()
    
    # Prints the average hits for a generation
    print(f"Average Hits: {sum(hits)/len(hits)}")

def run_neat(config_path):
    """
    Runs the NEAT algorithm
    """

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path) # Loads the config file

    p = neat.Population(config) # Creates a population

    # Load from a checkpoint, uncomment to use
    #p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-499")

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True)) 
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(10)) # Creates a checkpoint every n generations

    # Runs n generations
    winner = p.run(eval_genomes, 100)

    # Save the best genome to a file
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close

def test_neat(config_path, genome_path="winner.pkl"):
    """
    Run the best genome on the game
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path) # Loads the config file

    # Load the best genome from file
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Correctly formats the genome
    genomes = [(1, genome)]

    # Runs the game, in testing mode
    eval_genomes(genomes, config, True)

# Run the program if this is the main file, not an import
if __name__ == '__main__':
    # Get the path to the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    # Use the NEAT algorithm, uncomment one
    #run_neat(config_path) # Runs training
    test_neat(config_path) # Runs the best genome

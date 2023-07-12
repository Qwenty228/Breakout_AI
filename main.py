import pygame as pg
import random
from numpy import interp
import neat
import os


pg.init()
WIDTH, HEIGHT = SIZE = (800, 600)
SPEED = 5

COLORS = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
FPS = 60


class Block(pg.sprite.Sprite):
    def __init__(self, x, y, *groups: pg.sprite.Group) -> None:
        super().__init__(*groups)
        self.image = pg.Surface((60, 40))
        color_num = interp(x, [0, HEIGHT], [0, len(COLORS)-1])
        i, f = divmod(color_num, 1)
        color = pg.Color(COLORS[int(i)]).lerp(
            pg.Color(COLORS[min(int(i)+1, len(COLORS)-1)]), f)

        self.image.fill(color)

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def update(self, *arg, **kwarg) -> None:
        pass


class Ball(pg.sprite.Sprite):
    def __init__(self, x, y, color, *groups: pg.sprite.Group) -> None:
        super().__init__(*groups)
        self.image = pg.Surface((20, 20))
        self.image.fill(color)

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.dx_add = 0

        self.dx = -1*SPEED
        self.dy = random.choice([-1, 1]) * SPEED
        self.isOver = False

    def update(self, platform, blocks, genome=None) -> None:
        if self.rect.top <= 0:
            self.dy *= -1
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.dx *= -1
        if self.rect.bottom >= HEIGHT:
            self.isOver = True

        if self.rect.colliderect(platform.rect):
            self.dy *= -1
            self.dx += platform.vel
            self.rect.bottom = platform.rect.top

        for block in blocks:
            if self.rect.colliderect(block.rect):
                self.dy *= -1
                blocks.remove(block)
                genome.fitness += 10


        self.rect.x += self.dx + self.dx_add * (-1 if self.dx < 0 else 1)
        self.rect.y += self.dy


class Platform(pg.sprite.Sprite):
    def __init__(self, y, color, *groups: pg.sprite.Group) -> None:
        super().__init__(*groups)
        self.image = pg.Surface((100, 20))
        self.image.fill(color)

        self.rect = self.image.get_rect()
        self.rect.topleft = (int(WIDTH/2), y)

        self.vel = 0

    def update(self, *arg, **kwargs) -> None:
        if self.rect.left <= 0:
            self.rect.left = 0
        if self.rect.right >= WIDTH:
            self.rect.right = WIDTH

    def move(self, action):
        if action == -1:
            self.vel = - SPEED * 2
        elif action == 1:
            self.vel = SPEED * 2
        else:
            self.vel = 0
        
        self.rect.x += self.vel


def eval_genomes(genomes, config):
    packs = []
    for i, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        packs.append([genome, net, Sim(i)])

    screen = pg.display.get_surface()
    counter = len(packs)

    while True:
        clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
        
        screen.fill((0, 0, 0))

        count = counter
        for genome, net, sim in packs:
            if sim.ball.isOver:
                if Sim.draw_block == sim:
                    Sim.draw_block = None
                count -= 1
                continue
            
            if Sim.draw_block == None:
                Sim.draw_block = sim

            genome.fitness += 0.2
            dist = abs(sim.ball.rect.centerx - sim.platform.rect.centerx)
            if dist > 150:
                genome.fitness -= 0.001 * dist
            
            output = net.activate((sim.ball.rect.centerx, sim.ball.dy, abs(sim.ball.dx) + sim.ball.dx_add , sim.platform.rect.centerx))
            

            action = output.index(max(output)) - 1
            sim.platform.move(action)

            sim.sim_sprites.update(sim.platform, sim.blocks, genome=genome)
            win = sim.block_check()
            if win: 
                genome.fitness += 1000
                sim.ball.isOver = True
                count -= 1
                continue
           
            sim.sim_sprites.draw(screen)
            
            
        
        if Sim.draw_block:
            Sim.draw_block.blocks.draw(screen)

        if count == 0:
            break
      



        pg.display.flip()


    


class Sim:
    generated = {}
    draw_block = None

    def __new__(cls, i):
        if i not in cls.generated:
            new = super().__new__(cls)
            new.sim_sprites = pg.sprite.Group()
            color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            new.platform = Platform(HEIGHT*0.95, color,  new.sim_sprites)
            new.ball = Ball(WIDTH/2, HEIGHT*0.8, color, new.sim_sprites)
            new.blocks = None
            cls.generated[i] = new

        cls.generated[i].restart()
        return cls.generated[i]

    def restart(self):
        self.ball.rect.center = (WIDTH/2, HEIGHT*0.8)
        self.platform.rect.center = (WIDTH/2, HEIGHT*0.95)
        self.ball.isOver = False
        self.ball.dx_mult = 1
    
        self.blocks = pg.sprite.Group()
        for i in range(50, WIDTH, 70):
            for j in range(30, int(HEIGHT*0.6), 50):
                Block(i, j, self.blocks)

        self.block_amount = len(self.blocks)

    def block_check(self):
        if self.block_amount == 0:
            return True
        elif self.block_amount - 5 > len(self.blocks):
            self.ball.dx_add = 1 - (len(self.blocks) / self.block_amount)
            
            

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes)

    print("Best fitness -> {}".format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-FeedForward")
    screen = pg.display.set_mode(SIZE)
    pg.display.set_caption("Breakout")
    clock = pg.time.Clock()
    run(config_path)

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
    def __init__(self, x, y, *groups: pg.sprite.Group) -> None:
        super().__init__(*groups)
        self.all_sprites = groups[0]
        self.image = pg.Surface((20, 20))
        self.image.fill([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.dx = -1*SPEED
        self.dy = random.choice([-1, 1]) * SPEED
        self.isOver = False

    def update(self, platform_rect, blocks, genome=None) -> None:
        if self.rect.top <= 0:
            self.dy *= -1
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.dx *= -1
        if self.rect.bottom >= HEIGHT:
            self.isOver = True

        if self.rect.colliderect(platform_rect):
            self.dy *= -1
            self.rect.bottom = platform_rect.top

        for block in blocks:
            if self.rect.colliderect(block.rect):
                self.dy *= -1
                self.all_sprites.remove(block)
                blocks.remove(block)
                genome.fitness += 1


        self.rect.x += self.dx
        self.rect.y += self.dy


class Platform(pg.sprite.Sprite):
    def __init__(self, y, *groups: pg.sprite.Group) -> None:
        super().__init__(*groups)
        self.image = pg.Surface((100, 20))
        self.image.fill((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.topleft = (int(WIDTH/2), y)

    def update(self, *arg, **kwargs) -> None:
        if self.rect.left <= 0:
            self.rect.left = 0
        if self.rect.right >= WIDTH:
            self.rect.right = WIDTH

    def move(self, action):
        if action == -1:
            self.rect.x -= SPEED * 2
        elif action == 1:
            self.rect.x += SPEED * 2


def eval_genomes(genomes, config):
    packs = []
    for i, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        packs.append([net, genome, Sim(i)])


    first_sim = packs[0][2]

    surface = pg.display.get_surface()

    over = len(packs)

    while 1:
        Game.clock.tick(FPS)
        surface.fill((0, 0, 0))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        count = over
        for net, genome, sim in packs:
            if sim.ball.isOver:
                count -= 1
                continue
            genome.fitness += 0.1
            output = net.activate(
                (sim.ball.dx, sim.ball.dy, sim.ball.rect.centerx, sim.ball.rect.centery))
            sim.platform.move(output.index(max(output))-1)
            sim.update(genome)

        if count == 0:
            break
            
       

        first_sim.blocks.draw(surface)
        Game.draw_sprites.draw(surface)

        pg.display.update()


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


class Game:
    clock = pg.time.Clock()
    draw_sprites = pg.sprite.Group()

    def __init__(self) -> None:
        self.screen = pg.display.set_mode(SIZE)


class Sim:
    created = {}
    def __new__(cls, id) -> None:
        if (sim := Sim.created.get(id)) is not None:
            sim.restart()
            return sim

       

        sim = super().__new__(cls)
        

        sim.all_sprites = pg.sprite.Group()
        sim.blocks = pg.sprite.Group()
        for i in range(30, WIDTH, 60):
            for j in range(30, int(HEIGHT*0.6), 40):
                Block(i, j, sim.all_sprites, sim.blocks)

        sim.platform = Platform(
            HEIGHT*0.95, sim.all_sprites, Game.draw_sprites)
        sim.ball = Ball(WIDTH/2, HEIGHT*0.8,
                         sim.all_sprites, Game.draw_sprites)
        
        Sim.created[id] = sim

        
        return sim
        

        
        
    def restart(self):
        self.platform.rect.center = (int(WIDTH/2), HEIGHT*0.95)
        self.ball.rect.center = (WIDTH/2, HEIGHT*0.8)
        self.ball.isOver = False
        for block in self.blocks:
            self.all_sprites.remove(block)


        self.blocks = pg.sprite.Group()
        for i in range(30, WIDTH, 60):
            for j in range(30, int(HEIGHT*0.6), 40):
                Block(i, j, self.all_sprites, self.blocks)


    def update(self, genome):
        self.all_sprites.update(self.platform.rect, self.blocks, genome=genome)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-FeedForward")
    Game()
    run(config_path)

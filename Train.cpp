#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "net.h"
#include "snake.h"

// settings
const std::vector<int> LAYOUT = {5, 8, 3};
const Act ACTIVATION = Act::Tanh;
const int POP_SIZE = 200;
const int ELITE = 15;
const int TOURNAMENT = 3;
const int GAMES_EACH = 5;
const float MUT_RATE = 0.05f;
const float MUT_SIGMA = 0.3f;

struct Individual
{
    Net net;
    float fit = 0.f;
    explicit Individual(const Net &n) : net(n) {}
};

// Run game
int playGame(const Net &net, unsigned seed, float &fitOut)
{
    Snake s(seed);
    while (s.isAlive)
        s.step(Net::argmax(net.forward(s.inputs())));
    fitOut = s.score * 100.f + s.steps;
    return s.score;
}

// Avg fitness over several games, for more reliable selection.
float evaluate(const Net &net, unsigned firstSeed)
{
    float total = 0.f;
    for (int k = 0; k < GAMES_EACH; ++k)
    {
        float f;
        playGame(net, firstSeed + k, f);
        total += f;
    }
    return total / GAMES_EACH;
}

// Selection breeding
const Individual &pickParent(const std::vector<Individual> &pop, std::mt19937 &rng)
{
    const Individual *best = &pop[rng() % pop.size()];
    for (int i = 1; i < TOURNAMENT; ++i)
    {
        const Individual *c = &pop[rng() % pop.size()];
        if (c->fit > best->fit)
            best = c;
    }
    return *best;
}

// Average / best apples over many fresh games, for honest reporting.
void report(const char *label, const Net &net, int games, unsigned seedBase)
{
    double sum = 0;
    int best = 0;
    for (int i = 0; i < games; ++i)
    {
        float f;
        int apples = playGame(net, seedBase + i, f);
        sum += apples;
        best = std::max(best, apples);
    }
    std::cout << label << ": average " << sum / games << " apples, best " << best << "\n";
}

int main(int argc, char **argv)
{
    int generations = (argc > 1) ? std::atoi(argv[1]) : 100;
    std::mt19937 rng(2024);

    {
        double sum = 0;
        for (int i = 0; i < 300; ++i)
        {
            Net r(LAYOUT, ACTIVATION);
            r.randomize(rng);
            float f;
            sum += playGame(r, 900000 + i, f);
        }
        std::cout << "random brains: average " << sum / 300 << " apples\n\n";
    }

    // --- generation 0: random population ---
    std::vector<Individual> pop;
    for (int i = 0; i < POP_SIZE; ++i)
    {
        Net n(LAYOUT, ACTIVATION);
        n.randomize(rng);
        pop.emplace_back(n);
    }

    for (int gen = 0; gen < generations; ++gen)
    {
        unsigned seed = 1000u * (gen + 1);
        double sum = 0;
        for (Individual &ind : pop)
        {
            ind.fit = evaluate(ind.net, seed);
            sum += ind.fit;
        }

        std::sort(pop.begin(), pop.end(),
                  [](const Individual &a, const Individual &b)
                  { return a.fit > b.fit; });

        if (gen % 10 == 0 || gen == generations - 1)
            std::cout << "gen " << gen << "  best fitness " << pop[0].fit
                      << "  average " << sum / POP_SIZE << "\n";

        std::vector<Individual> next;
        for (int i = 0; i < ELITE; ++i)
            next.push_back(pop[i]);
        while ((int)next.size() < POP_SIZE)
        {
            const Individual &a = pickParent(pop, rng);
            const Individual &b = pickParent(pop, rng);
            Net child = Net::crossover(a.net, b.net, rng);
            child.mutate(rng, MUT_RATE, MUT_SIGMA);
            next.emplace_back(child);
        }
        pop = std::move(next);
    }

    for (Individual &ind : pop)
        ind.fit = evaluate(ind.net, 777000u);
    std::sort(pop.begin(), pop.end(),
              [](const Individual &a, const Individual &b)
              { return a.fit > b.fit; });
    const Net &champion = pop[0].net;

    std::cout << "\n";
    report("champion on 300 fresh games", champion, 300, 500000u);
    champion.save("best_snake.txt");
    std::cout << "saved to best_snake.txt\n";
}
#pragma once
#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

enum class Act
{
    Tanh,
    ReLU,
    LeakyReLU,
};

struct Net
{
    std::vector<int> sizes;
    std::vector<float> genome;

    Act act = Act::Tanh;

    static size_t count_genes(const std::vector<int> &s)
    {
        size_t cnt = 0;
        for (size_t i = 0; i + 1 < s.size(); ++i)
        {
            cnt += (size_t)(s[i] + 1) * s[i + 1];
        }
        return cnt;
    }

    explicit Net(std::vector<int> layerSizes, Act activation = Act::Tanh)
        : sizes(std::move(layerSizes)), act(activation)
    {
        genome.resize(count_genes(sizes), 0.0f);
    }

    void randomize(std::mt19937 &rng, float range = 1.0f)
    {
        std::uniform_real_distribution<float> dist(-range, range);
        for (float &g : genome)
        {
            g = dist(rng);
        }
    }

    // Forward pass
    std::vector<float> forward(const std::vector<float> &input) const
    {
        assert((int)input.size() == sizes[0]);
        std::vector<float> x = input;

        size_t p = 0;
        bool is_output_layer = false;

        for (size_t l = 0; l + 1 < sizes.size(); ++l)
        {
            const int incnt = sizes[l];
            const int outcnt = sizes[l + 1];
            is_output_layer = (l + 2 == sizes.size());

            std::vector<float> y(outcnt);
            for (int i = 0; i < outcnt; ++i)
            {
                float z = 0.0f;
                for (int j = 0; j < incnt; ++j)
                {
                    z += x[j] * genome[p++];
                }
                z += genome[p++]; // bias
                y[i] = is_output_layer ? z : activate(z);
            }
            x = y;
        }
        assert(p == genome.size());
        return x;
    }

    static int argmax(const std::vector<float> &v)
    {
        return (int)(std::max_element(v.begin(), v.end()) - v.begin());
    }

    // Mutate
    static Net crossover(const Net &a, const Net &b, std::mt19937 &rng)
    {
        assert(a.sizes == b.sizes);
        Net c = a;
        std::bernoulli_distribution dist(0.5);
        for (size_t i = 0; i < c.genome.size(); ++i)
        {
            if (dist(rng))
                c.genome[i] = b.genome[i];
        }
        return c;
    }

    void mutate(std::mt19937 &rng, float rate = 0.05f, float sigma = 0.2f)
    {
        std::normal_distribution<float> dist(0.0f, sigma);
        std::bernoulli_distribution mutate_dist(rate);
        for (size_t i = 0; i < genome.size(); ++i)
        {
            if (mutate_dist(rng))
            {
                genome[i] += dist(rng);
            }
        }
    }

    // Load & Save
    bool save(const std::string &path) const
    {
        std::ofstream f(path);
        if (!f)
            return false;
        f << sizes.size() << "\n";
        for (int s : sizes)
            f << s << " ";
        f << "\n"
          << (int)act << "\n";
        for (float g : genome)
            f << g << " ";
        return (bool)f;
    }

    static bool load(const std::string &path, Net &out)
    {
        std::ifstream f(path);
        if (!f)
            return false;
        size_t n;
        f >> n;
        std::vector<int> s(n);
        for (int &v : s)
            f >> v;
        int act;
        f >> act;
        Net net(s, (Act)act);
        for (float &g : net.genome)
            f >> g;
        if (!f)
            return false;
        out = net;
        return true;
    }

private:
    float activate(float z) const
    {
        switch (act)
        {
        case Act::Tanh:
            return std::tanh(z);
        case Act::ReLU:
            return std::max(0.f, z);
        case Act::LeakyReLU:
            return z > 0.f ? z : 0.01f * z;
        }
        return z;
    }
};

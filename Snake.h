#include <iostream>
#include <deque>
#include <vector>
#include <random>

const int DX[4] = {0, 1, 0, -1};
const int DY[4] = {-1, 0, 1, 0};

struct Pos
{
    int x, y;
};

class Snake
{
public:
    static constexpr int Width = 10;
    static constexpr int Height = 10;

    std::deque<Pos> body;
    Pos food;
    std::mt19937 rng;
    int dir = 0; // 0: up, 1: right, 2: down, 3: left
    int steps = 0;
    int stepsSinceFood = 0;
    int score = 0;
    bool isAlive = true;
    int maxStepsWithoutFood = 100;

    explicit Snake(unsigned seed) : rng(seed)
    {
        reset();
    }

    void reset()
    {
        body.clear();
        body.push_back({Width / 2, Height / 2});
        isAlive = true;
        dir = 0;
        steps = 0;
        stepsSinceFood = 0;
        score = 0;
        placeFood();
    }

    bool isBody(int x, int y) const
    {
        for (const auto &p : body)
        {
            if (p.x == x && p.y == y)
                return true;
        }
        return false;
    }

    bool isInDanger(int x, int y) const
    {
        return (x < 0 || x >= Width || y < 0 || y >= Height || isBody(x, y));
    }

    void placeFood()
    {
        std::vector<Pos> emptyCells;
        for (int y = 0; y < Height; ++y)
        {
            for (int x = 0; x < Width; ++x)
            {
                if (!isBody(x, y))
                    emptyCells.push_back({x, y});
            }
        }
        if (emptyCells.empty())
        {
            isAlive = false; // No space for food, game over
            return;
        }
        food = emptyCells[rng() % emptyCells.size()];
    }
    void step(int action)
    {
        if (!isAlive)
        {
            return;
        }
        if (action == 1) // turn right
        {
            dir = (dir + 1) % 4;
        }
        else if (action == 2) // turn left
        {
            dir = (dir + 3) % 4; // equivalent to (dir - 1 + 4) % 4
        }
        Pos head = {body.front().x + DX[dir], body.front().y + DY[dir]};
        steps++;
        stepsSinceFood++;
        if (isInDanger(head.x, head.y))
        {
            isAlive = false;
            return;
        }
        body.push_front(head);
        if (head.x == food.x && head.y == food.y)
        {
            score++;
            stepsSinceFood = 0;
            placeFood();
        }
        else
        {
            body.pop_back();
        }
        if (stepsSinceFood > maxStepsWithoutFood)
        {
            isAlive = false;
        }
    }
    std::vector<float> inputs() const
    {
        Pos h = body.front();
        int fwd = dir, left = (dir + 3) % 4, right = (dir + 1) % 4;
        auto danger = [&](int d)
        {
            return isInDanger(h.x + DX[d], h.y + DY[d]) ? 1.f : 0.f;
        };
        int fx = food.x - h.x, fy = food.y - h.y;
        float ahead = (float)(fx * DX[fwd] + fy * DY[fwd]);
        float side = (float)(fx * DX[right] + fy * DY[right]);
        return {danger(fwd), danger(left), danger(right), ahead / Width, side / Width};
    }

    // console renderer (handy for debugging without a window)
    void draw() const
    {
        for (int y = -1; y <= Height; ++y)
        {
            for (int x = -1; x <= Width; ++x)
            {
                if (x < 0 || x >= Width || y < 0 || y >= Height)
                    std::cout << '#';
                else if (x == body.front().x && y == body.front().y)
                    std::cout << 'O';
                else if (isBody(x, y))
                    std::cout << 'o';
                else if (x == food.x && y == food.y)
                    std::cout << '*';
                else
                    std::cout << ' ';
            }
            std::cout << '\n';
        }
    }
};
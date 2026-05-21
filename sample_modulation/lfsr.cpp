#include <iostream>
#include <string>
#include <bit>
#include <vector>
using ull = unsigned long long;

int main()
{
    ull n, t, N;
    std::cout << "ビット数n, タップ数t, ステップ数N をスペース区切りで入力してください: ";
    if (!(std::cin >> n >> t >> N)) return 0;

    std::cout << "初期状態のビット列（例: 1001）を入力してください: ";
    std::string input_bits;
    std::cin >> input_bits;

    ull state = 0;
    for (int i = 0; i < n; i++)
        state = (state << 1) | (input_bits[i] - '0');

    ull feedback_mask = 0;
    std::cout << t << " 個のタップ位置（0始まりのインデックス）をスペース区切りで入力してください: ";
    for (int i = 0; i < t; i++)
    {
        int k;
        std::cin >> k;
        feedback_mask |= (1ULL << k);
    }

    // LFSRの動作定義（バグ修正版）
    auto lfsr_step = [&](ull current_state) -> ull
    {
        // std::popcount を使用し、マスクのカッコを修正
        ull feedback = std::popcount(current_state & feedback_mask) % 2;
        current_state = ((current_state << 1) & ((1ULL << n) - 1)) | feedback;
        return current_state;
    };

    std::vector<ull> transitions(N, 0);
    transitions[0] = lfsr_step(state);

    for (int k = 1; k < N; k++)
        transitions[k] = lfsr_step(transitions[k - 1]);

    std::string res = "\n--- 遷移結果 ---\n";
    for (int k = 0; k < N; k++)
    {
        res += std::to_string(k + 1) + "番目の遷移は ";
        for (int i = n - 1; i >= 0; i--)
            res += ((transitions[k] & (1ULL << i)) ? '1' : '0');
        res += " です\n";
    }
    std::cout << res << std::endl;
}

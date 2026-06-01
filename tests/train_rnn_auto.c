#include "layers_prot.h"
#include "linear_layer.h"
#include "rnn_cell.h"
#include "sgd.h"
#include "tensor.h"
#include "var.h"
#include <math.h>
#include <stdio.h>

#define SEQ_LEN 50
#define EPOCHS 10
#define LR 0.01f

#define ALPHA 0.2f
#define HIDDEN_FEATURE 16
#define RNN_FEATURE 16
#define CHECKPOINT 50

int main() {
  // 1. Generate Data (Sine Wave)
  f32 data[SEQ_LEN];
  for (u32 i = 0; i < SEQ_LEN; ++i) {
    data[i] = sinf((f32)i * ALPHA);
  }

  // 2. Initialize Architecture
  // RNN: in_features=1, batch=1, hidden=16, out=16
  RNNCell rnn = rnn_cell_new(1, 1, HIDDEN_FEATURE, RNN_FEATURE);
  LinearLayer proj = linear_layer_new(RNN_FEATURE, 1);

  f32 rnn_w[] = {
      -0.00313079, -0.07111216, -0.21353775, 0.21777621,  -0.19633588,
      -0.12654725, 0.16040075,  -0.02997565, -0.14095566, -0.17709911,
      -0.0870842,  -0.13785431, -0.1643832,  0.00202003,  -0.00298843,
      0.09741217,  0.14434177,  0.0562101,   0.06469911,  -0.2003456,
      -0.16915265, 0.10901037,  -0.02558744, 0.23910424,  0.13902602,
      -0.1835846,  -0.07206479, -0.10120344, -0.10623875, -0.0618543,
      0.04571411,  0.11012754,  -0.21770045, 0.01207229,  0.11953771,
      0.11053377,  -0.16805753, 0.20977214,  -0.11920542, -0.1746693,
      -0.08682081, -0.24816874, 0.06760234,  -0.02792069, -0.0566892,
      -0.13656068, 0.11306939,  -0.24199954, 0.22866675,  0.11198509,
      -0.19018236, 0.03980699,  -0.20872098, 0.0116137,   -0.19336838,
      -0.13710666, 0.21045277,  -0.23911265, 0.2408081,   0.15986651,
      0.15787804,  0.1516076,   0.03390703,  0.06308854,  0.15468156,
      0.21858707,  -0.18991551, 0.2229416,   0.10830063,  -0.08984447,
      0.02919558,  0.06221148,  0.17673483,  0.04580715,  0.24880892,
      0.1064361,   0.16015616,  -0.16396534, -0.17489189, -0.01326695,
      0.10049334,  0.16379192,  -0.24206296, -0.15430665, 0.24707729,
      0.12890872,  -0.04646033, 0.15027007,  -0.12431711, -0.15893447,
      -0.20887974, 0.01609814,  -0.0668489,  0.24754873,  0.07397488,
      0.16387525,  -0.0172911,  0.08211547,  0.1119397,   -0.17879754,
      -0.1850012,  -0.08630118, 0.20814791,  0.0123848,   -0.02687177,
      0.0461587,   -0.01966316, 0.21742627,  0.09553891,  0.16139892,
      0.15819827,  0.08486348,  -0.18068635, 0.16994336,  0.17153001,
      -0.24904257, -0.09278759, -0.16036618, 0.11657378,  0.2379365,
      -0.1461415,  -0.0860714,  0.16652727,  0.14908975,  0.03768852,
      0.13496011,  0.13843903,  0.04645544,  -0.16542548, 0.01153612,
      0.00445047,  -0.05309442, 0.14093193,  0.19164205,  -0.11221528,
      0.1133073,   0.05991241,  0.0766474,   0.07155669,  0.01841569,
      -0.1996792,  0.18264705,  0.01429948,  0.23535663,  -0.15649047,
      0.05484396,  0.05204171,  -0.14966187, 0.0389201,   0.16258362,
      -0.07970646, 0.20487052,  0.10071796,  -0.14131668, 0.1808621,
      0.10246566,  0.07560852,  -0.09420705, 0.22816104,  -0.19374377,
      0.02787992,  -0.1869812,  0.00137973,  -0.19268465, -0.07618055,
      0.22088075,  -0.01886261, -0.15070623, -0.05832374, 0.08083406,
      0.22450444,  -0.07534462, -0.12472209, 0.03797397,  0.10357815,
      -0.12258324, 0.19511357,  0.06899172,  -0.14991558, 0.23442677,
      0.08831102,  0.19619784,  0.22514617,  -0.19410142, -0.14864889,
      -0.1199528,  -0.06862858, -0.18544966, 0.11528391,  -0.17248562,
      0.09113187,  -0.11035487, -0.00478834, 0.07323971,  0.03839597,
      -0.07948747, -0.18383881, 0.02588418,  -0.038046,   -0.06424809,
      0.20804954,  0.16879812,  -0.02139127, 0.02318785,  -0.15115643,
      0.15229717,  0.06215242,  -0.05276984, -0.01201421, 0.0307945,
      -0.15124759, -0.01632825, -0.09136674, -0.07790214, -0.1710439,
      0.00578958,  -0.1822243,  0.10381323,  0.1864998,   -0.05633196,
      -0.16617155, 0.11199427,  0.14946127,  0.00612098,  0.13187736,
      0.10260525,  -0.09010759, 0.13527468,  0.0675942,   0.0204961,
      -0.22045854, -0.02670336, -0.22616073, -0.19215792, -0.0982801,
      0.15228128,  0.09255019,  -0.1816811,  -0.09909564, 0.1107946,
      -0.16891795, 0.09307823,  -0.04991364, 0.09234291,  0.03391373,
      0.10532203,  0.12071374,  -0.20702067, 0.08920541,  0.08150581,
      0.182518,    0.07561332,  0.16058916,  -0.03273734, 0.02016154,
      0.03991967,  -0.17281663, -0.07512286, 0.19566241,  -0.08455873,
      0.01742309,  -0.23250282, -0.04235536, -0.23756653, 0.22795439,
      -0.08612025, 0.05782518,  0.06952989,  0.17122331,  0.2137287,
      -0.2353906,  0.0662829};
  f32 rnn_b[] = {0.24682117,  -0.06313074, 0.10570627,  -0.2268253,
                 -0.22123015, 0.04459384,  -0.1487535,  -0.01619205,
                 -0.00119537, 0.13320339,  -0.01833886, 0.08554783,
                 0.23866606,  -0.10985404, -0.18978873, 0.2453697};
  f32 ll_w[] = {0.19754761,  -0.06212181, -0.12588847, -0.14880869,
                -0.14969695, -0.20253205, 0.08047888,  0.12206304,
                -0.05988231, 0.07485434,  -0.18517804, 0.06303057,
                -0.02379316, -0.12688255, 0.22718042,  0.18483934};
  f32 ll_b[] = {0.21410999};

  for (u32 i = 0; i < rnn->cell.weight->data->nelements; ++i) {
    rnn->cell.weight->data->data[i] = rnn_w[i];
  }
  for (u32 i = 0; i < rnn->cell.bias->data->nelements; ++i) {
    rnn->cell.bias->data->data[i] = rnn_b[i];
  }
  for (u32 i = 0; i < proj->weight->data->nelements; ++i) {
    proj->weight->data->data[i] = ll_w[i];
  }
  for (u32 i = 0; i < proj->bias->data->nelements; ++i) {
    proj->bias->data->data[i] = ll_b[i];
  }

  rnn_cell_track(rnn);
  linear_layer_track(proj);

  // 3. Initialize Optimizer
  Tensor params[] = {rnn->cell.weight, rnn->cell.bias, proj->weight,
                     proj->bias};
  SgdOptimizer optim = sgd_optimizer_new(params, 4, LR);

  // 4. Training Loop
  for (u32 epoch = 0; epoch < EPOCHS; ++epoch) {

    // CRITICAL: Sever the BPTT graph from the previous epoch
    Tensor init_h = tensor_zero(2, (u32[]){1, HIDDEN_FEATURE});
    rnn->hidden_state = track_replace_untracked(init_h);

    // Initialize sequence loss accumulator
    Tensor total_loss = track(tensor_scalar(0.0f));

    for (u32 time = 0; time < SEQ_LEN - 1; ++time) {
      // Get current and next time step
      Tensor x_t = tensor_new(2, (u32[]){1, 1});
      tensor_set(x_t, (u32[]){0, 0}, data[time]);

      Tensor y_true = tensor_new(2, (u32[]){1, 1});
      tensor_set(y_true, (u32[]){0, 0}, data[time + 1]);

      // Forward Pass
      Tensor pre_act = rnn_cell_forward(rnn, x_t);
      Tensor y_pred = linear_layer_forward(proj, rnn->hidden_state);

      // if(epoch==0 ){

      //}

      // Compute MSE Step Loss: (y_pred - y_true)^2
      Tensor diff = var_sub(y_pred, y_true);
      Tensor sq_err = var_mul(diff, diff);

      // Accumulate total loss for the sequence
      Tensor next_loss = var_add(total_loss, sq_err);

      // Graph cleanup for intermediate nodes (if your engine requires it)
      // The DAG handles the memory, but pointers move.
      total_loss = next_loss;
    }

    // Backward Pass
    var_backward(total_loss);

    // Optimizer Step
    Tensor grads[] = {var_grad(params[0]), var_grad(params[1]),
                      var_grad(params[2]), var_grad(params[3])};
    sgd_optimize(optim, grads);

    // Print & Zero Grads
    // if (epoch % CHECKPOINT == 0)
    {
      printf("Epoch %3u | Loss: %.4f\n", epoch,
             tensor_get(total_loss, (u32[MAX_DIMS]){0}));
    }

    var_zero_grad(total_loss);
  }

  // 5. Cleanup
  sgd_optimizer_destroy(optim);
  rnn_cell_destroy(rnn);
  linear_layer_destroy(proj);

  return 0;
}

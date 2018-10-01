if itr % self.target_update_freq == 0:
          self.target_q_network.set_weights(
            self.q_network.get_weights())

        if itr % self.train_freq == 0:
          loss = self.update_policy(itr)


update_pollicy:

target_batch = self.q_network.predict(self.input_batch) ### LIVE GET TARGER FIRST
nextstate_q_values = self.target_q_network.predict(self.nextstate_batch) ## TARGET GET NEXTSTATE TARGETS
if self.learning_type == _LEARNING_TYPE_DOUBLE:
    nextstate_q_values_live_network = self.q_network.predict(self.nextstate_batch) ## LIVE GET NEXTSTATE
for i in range(ns):
    # to incur 0 loss on all actions but the one we care about,...
    # target_batch[i, ...] = cur_q_values[i, ...]
    _, action, reward, _, is_terminal = samples[i]
    if is_terminal:
        target_batch[i, action] = reward ### IF is_terminal ONLY REWARD!
    else:
        if self.learning_type == _LEARNING_TYPE_DOUBLE:
            selected_action = np.argmax(nextstate_q_values_live_network[i].flatten()) # FIND THE ACTION BASED ON LIVE_NEXTSTATE
            target_batch[i, action] = reward + self.gamma * nextstate_q_values[i, selected_action] # USE TARGET_NETWORK VALUES FOR UPDATE
    else:
        target_batch[i, action] = reward + self.gamma * np.max(nextstate_q_values[i])

self.training_reward_seen += sum([el[2] for el in samples])


loss = self.q_network.train_on_batch(self.input_batch, target_batch)

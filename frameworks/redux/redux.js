const createStore = (reducer, initialState) => {
  return {
    state: initialState,
    listeners: [],
    getState: () => {
      return store.state;
    },
    subscribe: (listener) => {
      store.listeners.push(listener);
    },
    dispatch: (action) => {
      console.log("> Action", action);
      store.state = reducer(store.state, action);
      store.listeners.forEach((listener) => listener());
    },
  };
};

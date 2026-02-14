export function combineReducers(reducers) {
  return function (state = {}, action, args) {
    return Object.entries(reducers).reduce(
      (acc, [name, reducer]) =>
        Object.assign(acc, {
          [name]: reducer(state[name], action, args),
        }),
      state
    );
  };
}

export function logger(reducer) {
  return function (prevState, action, args) {
    console.group(action);
    console.log("Previous State", prevState);
    console.log("Action Arguments", args);
    const next_state = reducer(prevState, action, args);
    console.log("Next State", next_state);
    console.groupEnd();
    return next_state;
  };
}

export function html([first, ...strings], ...values) {
  return values
    .reduce((acc, cur) => acc.concat(cur, strings.shift()), [first])
    .filter((x) => (x && x !== true) || x === 0)
    .join("");
}

export function createStore(reducer) {
  let state = reducer();
  const roots = new Map();

  const render = () => {
    for (const [r, component] of roots) {
      const output = component();
      r.innerHTML = output;
    }
  };

  return {
    attach(component, r) {
      roots.set(r, component);
      render();
    },
    connect(component, selector = (state) => state) {
      return (props, ...args) =>
        component({ ...props, ...selector(state) }, ...args);
    },
    dispatch(action, ...args) {
      state = reducer(state, action, args);
      render();
    },
    getState() {
      return state;
    },
  };
}

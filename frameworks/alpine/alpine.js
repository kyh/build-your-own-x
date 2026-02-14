const observe = (data, onChange) => {
  return new Proxy(data, {
    set(target, key, value) {
      target[key] = value;
      onChange();
      return true;
    },
    deleteProperty(target, key, onChange) {
      delete target[key];
      onChange();
      return true;
    },
  });
};

const walkDom = (el, callback) => {
  callback(el);
  el = el.firstElementChild;
  while (el) {
    walkDom(el, callback);
    el = el.nextElementSibling;
  }
};

const bootstrap = () => {
  const rootEls = document.querySelectorAll("[x-data]");
  rootEls.forEach((el) => {
    if (!el.__x) {
      try {
        el.__x = new Component(el);
      } catch (error) {
        setTimeout(() => {
          throw error;
        }, 0);
      }
    }
  });
};

const directives = {
  "x-text": (el, component, expression) => {
    const value = component.safeEval(expression);
    el.innerText = value;
  },
  "x-show": (el, component, expression) => {
    const value = component.safeEval(expression);
    el.style.display = value ? "block" : "none";
  },
};

class Component {
  constructor(el) {
    this.rootEl = el;
    this.rawData = this.getComponentData();
    this.$data = observe(this.rawData, this.refreshDOM.bind(this));
    this.initializeElement();
  }

  initializeElement() {
    this.registerListeners();
    this.refreshDOM();
  }

  getComponentData() {
    const dataString = this.rootEl.getAttribute("x-data");
    return Function(`"use strict";return (${dataString})`)();
  }

  registerListeners() {
    walkDom(this.rootEl, (el) => {
      Array.from(el.attributes).forEach((attribute) => {
        if (!attribute.name.startsWith("x-on:")) return;
        const event = attribute.name.replace("x-on:", "");
        el.addEventListener(event, () => {
          this.safeEval(attribute.value);
        });
      });
    });
  }

  refreshDOM() {
    walkDom(this.rootEl, (el) => {
      Array.from(el.attributes).forEach((attribute) => {
        if (!directives.hasOwnProperty(attribute.name)) return;
        directives[attribute.name](el, this, attribute.value);
      });
    });
  }

  safeEval(expression) {
    return Function(`with (this.$data) { return ${expression} }`).bind(this)();
  }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", bootstrap);

// Utilities
const equals = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const clone = (a) => {
  try {
    return JSON.parse(JSON.stringify(a));
  } catch (e) {
    return undefined;
  }
};

class Scope {
  static counter = 0;

  constructor(parent, id) {
    this.$$watchers = [];
    this.$$children = [];
    this.$parent = parent;
    this.$id = id || 0;
  }

  $watch(exp, fn) {
    this.$$watchers.push({
      exp,
      fn,
      last: clone(this.$eval(exp)),
    });
  }

  /**
   * In the complete implementation there's the lexer, parser,
   * and interpreter.
   */
  $eval(exp) {
    let val;
    if (typeof exp === "function") {
      val = exp.call(this);
    } else {
      try {
        val = Function(`with (this) { return eval(${exp}) }`).bind(this)();
      } catch (e) {
        val = undefined;
      }
    }
    return val;
  }

  $new() {
    Scope.counter += 1;
    const obj = new Scope(this, Scope.counter);
    Object.setPrototypeOf(obj, this);
    this.$$children.push(obj);
    return obj;
  }

  $destroy() {
    const pc = this.$parent.$$children;
    pc.splice(pc.indexOf(this), 1);
  }

  $digest() {
    let dirty;
    let watcher;
    let current;
    let i;

    do {
      dirty = false;
      for (i = 0; i < this.$$watchers.length; i += 1) {
        watcher = this.$$watchers[i];
        current = this.$eval(watcher.exp);
        if (!equals(watcher.last, current)) {
          watcher.last = clone(current);
          dirty = true;
          watcher.fn(current);
        }
      }
    } while (dirty);
    for (i = 0; i < this.$$children.length; i += 1) {
      this.$$children[i].$digest();
    }
  }
}

const Provider = {
  get(name, locals) {
    if (this._cache[name]) {
      return this._cache[name];
    }
    const provider = this._providers[name];
    if (!provider || typeof provider !== "function") {
      return null;
    }
    return (this._cache[name] = this.invoke(provider, locals));
  },
  directive(name, fn) {
    this._register(name + Provider.DIRECTIVES_SUFFIX, fn);
  },
  controller(name, fn) {
    this._register(name + Provider.CONTROLLERS_SUFFIX, () => fn);
  },
  service(name, fn) {
    this._register(name, fn);
  },
  annotate(fn) {
    const res = fn
      .toString()
      .replace(/((\/\/.*$)|(\/\*[\s\S]*?\*\/))/gm, "")
      .match(/\((.*?)\)/);
    if (res && res[1]) {
      return res[1].split(",").map((d) => d.trim());
    }
    return [];
  },
  invoke(fn, locals) {
    locals = locals || {};
    const deps = this.annotate(fn).map((s) => {
      return locals[s] || this.get(s, locals);
    });
    return fn.apply(null, deps);
  },
  _register(name, service) {
    this._providers[name] = service;
  },
  _cache: { $rootScope: new Scope() },
  _providers: {},
  DIRECTIVES_SUFFIX: "Directive",
  CONTROLLERS_SUFFIX: "Controller",
};

const DOMCompiler = {
  bootstrap() {
    this.compile(document.children[0], Provider.get("$rootScope"));
  },
  callDirectives(dom, $scope) {
    let isCreated = false;
    [...dom.attributes]
      .map((attr) => {
        const directive = Provider.get(attr.name + Provider.DIRECTIVES_SUFFIX);
        return (
          directive && {
            expr: attr.value,
            provision: directive,
          }
        );
      })
      .filter(Boolean)
      .forEach((d) => {
        if (d.provision.scope && !isCreated) {
          isCreated = true;
          $scope = $scope.$new();
        }
        d.provision.link(dom, $scope, d.expr);
      });
    return $scope;
  },
  compile(dom, scope) {
    scope = this.callDirectives(dom, scope);
    [...dom.children].forEach((d) => this.compile(d, scope));
  },
};

// Directives
Provider.directive("ng-bind", () => {
  return {
    scope: false,
    link: (el, scope, exp) => {
      el.innerHTML = scope.$eval(exp);
      scope.$watch(exp, (val) => {
        el.innerHTML = val;
      });
    },
  };
});

Provider.directive("ng-click", () => {
  return {
    scope: false,
    link: (el, scope, exp) => {
      el.onclick = () => {
        scope.$eval(exp);
        scope.$digest();
      };
    },
  };
});

Provider.directive("ng-show", () => {
  return {
    scope: false,
    link: (el, scope, exp) => {
      el.style.display = scope.$eval(exp) ? "block" : "none";
      scope.$watch(exp, (val) => {
        el.style.display = val ? "block" : "none";
      });
    },
  };
});

Provider.directive("ng-controller", () => {
  return {
    scope: true,
    link: (_el, scope, exp) => {
      const ctrl = Provider.get(exp + Provider.CONTROLLERS_SUFFIX);
      Provider.invoke(ctrl, { $scope: scope });
    },
  };
});

Provider.directive("ng-model", () => {
  return {
    link: (el, scope, exp) => {
      el.onkeyup = () => {
        scope[exp] = el.value;
        scope.$digest();
      };
      scope.$watch(exp, (val) => {
        el.value = val;
      });
    },
  };
});

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => DOMCompiler.bootstrap());

# Repilot Patch

```
return false;
```

# Developer Patch

```
        return false;
```

# Context

```
--- bug/Closure-86/src/com/google/javascript/jscomp/NodeUtil.java

+++ fix/Closure-86/src/com/google/javascript/jscomp/NodeUtil.java

@@ -2462,7 +2462,7 @@

         // TODO(nicksantos): This needs to be changed so that it
         // returns true iff we're sure the value was never aliased from inside
         // the constructor (similar to callHasLocalResult)
-        return true;
+return false;
       case Token.FUNCTION:
       case Token.REGEXP:
       case Token.ARRAYLIT:
```

# Note


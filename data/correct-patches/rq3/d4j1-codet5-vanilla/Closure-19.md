# Repilot Patch

```


      case Token.THIS:
        break;
```

# Developer Patch

```
      case Token.THIS:
        break;
```

# Context

```
--- bug/Closure-19/src/com/google/javascript/jscomp/type/ChainableReverseAbstractInterpreter.java

+++ fix/Closure-19/src/com/google/javascript/jscomp/type/ChainableReverseAbstractInterpreter.java

@@ -169,7 +169,10 @@

         scope.inferQualifiedSlot(node, qualifiedName, origType, type);
         break;
 
-        // "this" references aren't currently modeled in the CFG.
+
+
+      case Token.THIS:
+        break;
 
       default:
         throw new IllegalArgumentException("Node cannot be refined. \n" +
```

# Note


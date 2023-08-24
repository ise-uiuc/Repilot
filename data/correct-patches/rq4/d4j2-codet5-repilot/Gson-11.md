# Repilot Patch

```

      case STRING:
```

# Developer Patch

```
      case STRING:
```

# Context

```
--- bug/Gson-11/gson/src/main/java/com/google/gson/internal/bind/TypeAdapters.java

+++ fix/Gson-11/gson/src/main/java/com/google/gson/internal/bind/TypeAdapters.java

@@ -368,6 +368,8 @@

         in.nextNull();
         return null;
       case NUMBER:
+
+      case STRING:
         return new LazilyParsedNumber(in.nextString());
       default:
         throw new JsonSyntaxException("Expecting number, got: " + jsonToken);
```

# Note

